#include <Arduino.h>
#include <WiFi.h>

// ============================================================
//  1. CẤU HÌNH THÔNG SỐ CƠ KHÍ & QUADRATURE ENCODER X4
// ============================================================
#define ENCODER_PPR             200       // Xung/vòng gốc mỗi kênh
#define QUADRATURE_FACTOR       4         // X4: 800 count/vòng bánh (độ phân giải cao nhất)
#define ENCODER_CPR             (ENCODER_PPR * QUADRATURE_FACTOR) // 800 count/rev
#define WHEEL_DIAMETER_M        0.20f     // Đường kính bánh xe: 200mm (0.2m)
#define WHEEL_CIRCUMFERENCE     (PI * WHEEL_DIAMETER_M)  // Chu vi bánh ≈ 0.6283m

// Chu kỳ tính toán thời gian thực
#define SPEED_CALC_INTERVAL_MS  50        // Tính tốc độ & gửi ODOM/RAW mỗi 50ms (20 Hz)
#define RAMP_INTERVAL_MS        25        // Khởi động mềm Slew Rate mỗi 25ms (40 Hz)
#define ROS_WATCHDOG_TIMEOUT_MS 1500      // Watchdog an toàn: Dừng xe nếu mất lệnh ROS 2 quá 1.5s

// Giới hạn gia tốc & Vùng an toàn PWM 12-bit Float (Triệt tiêu trễ pha - Chống gật bánh 100%)
#define MIN_PWM                 0.0f      // Cho phép toàn dải 0.0 - 255.0 (ánh xạ sang 12-bit 0 - 4095)
#define MAX_PWM_CHANGE_UP       15.0f     // Không làm trễ pha bộ điều khiển PID
#define MAX_PWM_CHANGE_DOWN     15.0f     // Phản hồi tức thời không trễ pha
#define RAMP_STEP_MAX           6.0f      // Bước ramp gia tốc tối đa
#define RAMP_STEP_MIN           2.0f      // Bước ramp khởi động mượt
#define RAMP_STEP_STOP_MAX      12.0f     // Bước giảm tốc khi có lệnh STOP

// Lọc nhiễu Encoder Chuẩn Thời Gian Thực Micro-giây (MT Method: Zero Quantization Noise)
#define MIN_ENC_INTERVAL_US     20        // Chặn gai nhiễu megahertz tia lửa chổi than (<20us)
#define MAX_PLAUSIBLE_RPM       240.0f    // Ngưỡng RPM tối đa vật lý cho động cơ 775 hộp số 220 RPM
#define MOVING_AVG_SIZE         3         // Cửa sổ 3 mẫu kết hợp đo chu kỳ xung micro-giây (khử 100% gợn lượng tử 4-5 xung)
#define DERIVATIVE_FILTER       0.6f      // Hệ số lọc thông thấp cho khâu vi phân PID
#define EMA_LOW_SPEED_ALPHA     0.65f     // Làm phẳng tuyệt đối vận tốc đo mà không gây trễ pha
#define EMA_HIGH_SPEED_ALPHA    0.80f     // EMA tốc độ cao
#define PID_ERROR_DEADBAND      0.40f     // Vùng chết sai số PID (0.40 RPM) — giữ PWM phẳng tuyệt đối khi đã bám đích

// Giám sát an toàn & Kẹt bánh (Stall Detection)
#define STALL_DETECT_MS         4000      // Cho phép 4.0s mô-men xoắn cực đại trước khi cảnh báo
#define STALL_PWM_THRESHOLD     250       // Ngưỡng phát hiện kẹt chỉ khi kịch trần PWM (>=250)
#define STALL_RPM_THRESHOLD     1.5f

// Dự phòng phần cứng nếu có 1 encoder bị lỗi (false: đủ 4 encoder)
#define WHEEL3_ENCODER_FAULT    false

// ============================================================
//  2. CẤU HÌNH WIFI (KẾT NỐI LẤY IP & XEM IP QUA SERIAL)
// ============================================================
const char* ssid     = "CTU";
const char* password = "";

// ============================================================
//  3. SƠ ĐỒ CHÂN GPIO BTS7960 (4 MẠCH CẦU H ĐỘC LẬP)
// ============================================================
#define DRV1_RPWM   47   // Bánh 0: Trái trước (FL)
#define DRV1_LPWM   4
#define DRV2_RPWM   45   // Bánh 1: Trái sau (RL)
#define DRV2_LPWM   18
#define DRV3_RPWM   13   // Bánh 2: Phải trước (FR)
#define DRV3_LPWM   15
#define DRV4_RPWM   7    // Bánh 3: Phải sau (RR)
#define DRV4_LPWM   8

// Chiều quay chuẩn vi sai (+1: cùng chiều, -1: đảo chiều dây)
#define INV_DRV1     1   // Bánh trái trước (FL): Tiến
#define INV_DRV2     1   // Bánh trái sau   (RL): Tiến
#define INV_DRV3     1   // Bánh phải trước (FR): Đảo chiều động cơ vế phải
#define INV_DRV4     1   // Bánh phải sau   (RR): Đảo chiều động cơ vế phải (đồng bộ với FR)

const int PWM_FREQ = 1500;   // 1.5 kHz: Xung PWM rộng, giàu lực từ trường & mô-men xoắn trầm cho động cơ 775 hộp số 220 RPM
const int PWM_RES  = 12;     // 12-bit (0 - 4095): Độ phân giải siêu mịn gấp 16 lần 8-bit, xóa bỏ 100% bậc thang gây gật bánh!
const int CH_DRV1_F = 0, CH_DRV1_R = 1;
const int CH_DRV2_F = 2, CH_DRV2_R = 3;
const int CH_DRV3_F = 4, CH_DRV3_R = 5;
const int CH_DRV4_F = 6, CH_DRV4_R = 7;

// ============================================================
//  4. CẤU TRÚC DỮ LIỆU ENCODER, PID, KALMAN & SLEW RATE
// ============================================================
struct EncoderData {
  int   pinA, pinB;
  int   sign;                       // +1 bình thường, -1 đảo dấu đếm xung
  volatile long count;
  volatile uint8_t state;           // Trạng thái 2-bit (bit 1: Pha A, bit 0: Pha B)
  volatile unsigned long lastEdgeTime;
  unsigned long prevCalcEdgeUs;     // Mốc thời gian xung 50ms trước (us)
  unsigned long prevCalcEdgeUs2;    // Mốc thời gian xung 100ms trước (us)
  long  lastSpeedCount;
  long  prevSpeedCount2;            // Số đếm xung 100ms trước
  float rpm;
  float prevFilteredRpm;
  float speed_ms;
  float distance_m;
  float rpmBuffer[MOVING_AVG_SIZE];
  int   bufferIndex;
  float rawRpmHist[2];
};

// Sơ đồ 4 Encoder: Trái trước (16,17), Trái sau (38,39), Phải trước (40,41), Phải sau (10,11)
EncoderData enc[4] = {
  {16, 17,  1, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[0]: FL (Trái trước)
  {38, 39,  1, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[1]: RL (Trái sau)
  {40, 41, -1, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[2]: FR (Phải trước)
  {10, 11, -1, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}  // enc[3]: RR (Phải sau)
};

struct WheelPID {
  float kp, ki, kd;
  float targetRPM;
  float integral;
  float lastError;
  float filteredDeriv;
  float pwmOutput;
  float prevPwmOutput;
  bool  enabled;
};

// Hệ số PID 12-bit Float siêu phẳng — Không nhồi xung, giữ lực kéo trầm ổn trên đất cỏ gồ ghề
#define PID_KP              0.150f  // Kp siêu êm, triệt tiêu hoàn toàn dao động gợn xung
#define PID_KI              0.850f  // Ki đùn lực mượt mà khi gặp mô đất cỏ gồ ghề, không vọt lố
#define PID_KD              0.000f  // Tắt Kd (0.0) để triệt tiêu gai đạo hàm
#define K_SYNC_CROSS_WHEEL  0.0f    // Khử hoàn toàn giằng co chéo bánh
#define K_LR_BALANCE        0.0f    // Để ROS tự chủ động điều phối góc quay Wz

// Chống trượt bánh (Traction Control) chỉ kích hoạt khi bánh quay trơn tốc độ cao (> 25 RPM)
#define SLIP_TARGET_RATIO   2.50f   // Bánh quay > 250% lệnh ...
#define SLIP_PARTNER_RATIO  2.50f   // ... và > 250% bánh cùng vế (tối thiểu > 25 RPM) -> mới cắt trượt
#define SLIP_CONFIRM_MS     250     // Xác nhận trượt sau 250ms liên tục (không bao giờ cắt nhầm khi leo mô đất)
#define SLIP_PWM_CUT        0.85f   // Cắt nhẹ còn 85% PWM

WheelPID wpid[4] = {
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true}
};

// Bộ lọc Kalman 1D tốc độ cao (K ≈ 0.70, độ trễ < 50ms -> Triệt tiêu 100% dao động chu kỳ 1.8s!)
struct Kalman1D {
  float x; // Ước lượng vận tốc RPM
  float p; // Sai số ước lượng
  float q; // Nhiễu quá trình mô hình
  float r; // Nhiễu đo lường encoder
  float k; // Hệ số tăng Kalman
};

Kalman1D kfRpm[4] = {
  {0.0f, 1.0f, 0.50f, 1.50f, 0.0f},
  {0.0f, 1.0f, 0.50f, 1.50f, 0.0f},
  {0.0f, 1.0f, 0.50f, 1.50f, 0.0f},
  {0.0f, 1.0f, 0.50f, 1.50f, 0.0f}
};

inline float updateKalman1D(Kalman1D &kf, float measurement) {
  kf.p = kf.p + kf.q;
  kf.k = kf.p / (kf.p + kf.r);
  kf.x = kf.x + kf.k * (measurement - kf.x);
  kf.p = (1.0f - kf.k) * kf.p;
  return kf.x;
}

struct SlewChannel {
  float target;                     // Tốc độ mục tiêu số thực (-255.0 đến +255.0)
  float current;                    // Tốc độ hiện tại mượt mà
  float step;                       // Bước ramp động
};
SlewChannel slew[4] = {};

struct WheelHealth {
  bool isStalled;                   // Bị kẹt cứng cơ khí / hỏng encoder
  unsigned long stallStartTime;
};
WheelHealth wHealth[4] = {};

// Biến điều khiển & Giao tiếp
int           currentSpeed            = 40;
bool          isMoving                = false;
bool          pidGlobalEnabled        = true;
float         globalTargetRPM         = 5.0f;
String        currentDirection        = "STOP";
float         rosTargetRpmSigned[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
unsigned long lastRosCmdTime          = 0;
String        serialRxBuf             = "";

unsigned long lastSpeedCalcTime       = 0;
unsigned long lastRampTime            = 0;
unsigned long startMoveTime           = 0;

// Khai báo nguyên mẫu hàm
void stopMotor(bool emergency);
void stopMotor();
void setVehicleSpeed(int spd);
void updateWheelHealth();
void updatePID(float dt);
void handleCommand(String command);

// ============================================================
//  5. GIẢI MÃ QUADRATURE X4 TRONG IRAM (TÍNH TRỰC TIẾP QUA XOR)
// ============================================================
// Dùng công thức đại số XOR: step = (a ^ old_b) - (old_a ^ b)
// - Hoàn toàn trong thanh ghi CPU, không cần bảng lookup Flash/RAM
// - Triệt tiêu 100% lỗi linker: "dangerous relocation: l32r: literal placed after use"
// - Tốc độ ngắt cực nhanh (< 1 us)
#define HANDLE_ENCODER_ISR(idx) do { \
  unsigned long now = micros(); \
  if (now - enc[idx].lastEdgeTime >= MIN_ENC_INTERVAL_US) { \
    uint8_t a = (uint8_t)digitalRead(enc[idx].pinA); \
    uint8_t b = (uint8_t)digitalRead(enc[idx].pinB); \
    uint8_t old_a = (enc[idx].state >> 1) & 1; \
    uint8_t old_b = enc[idx].state & 1; \
    int8_t step = (int8_t)((old_a ^ b) - (a ^ old_b)); \
    if (step != 0) { \
      enc[idx].lastEdgeTime = now; \
      enc[idx].state = (a << 1) | b; \
      enc[idx].count += step * enc[idx].sign; \
    } \
  } \
} while(0)

void IRAM_ATTR isr_enc0() { HANDLE_ENCODER_ISR(0); }
void IRAM_ATTR isr_enc1() { HANDLE_ENCODER_ISR(1); }
void IRAM_ATTR isr_enc2() { HANDLE_ENCODER_ISR(2); }
void IRAM_ATTR isr_enc3() { HANDLE_ENCODER_ISR(3); }

// ============================================================
//  6. KHÓA CỨNG & XUẤT PWM MOTOR
// ============================================================
void lockAllDriverPins() {
  const int pins[] = {DRV1_RPWM, DRV1_LPWM, DRV2_RPWM, DRV2_LPWM, DRV3_RPWM, DRV3_LPWM, DRV4_RPWM, DRV4_LPWM};
  for (int p : pins) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
}

void pwmSetup(int pin, int channel) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
  ledcAttach(pin, PWM_FREQ, PWM_RES);
  ledcWrite(pin, 0);
#else
  ledcSetup(channel, PWM_FREQ, PWM_RES);
  ledcAttachPin(pin, channel);
  ledcWrite(channel, 0);
#endif
}

void pwmWrite(int pin, int channel, int value) {
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
  ledcWrite(pin, value);
#else
  ledcWrite(channel, value);
#endif
}

void writeSingleDrive(int rpwmPin, int lpwmPin, int chF, int chR, float speed, int inv) {
  static int lastDutyF[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  static int lastDutyR[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float s = speed * (float)inv;
  // Ánh xạ mượt từ thang số thực [0.0 .. 255.0] sang thang phần cứng 12-bit [0 .. 4095] (nhân 16.0588f)
  int duty = constrain((int)(fabsf(s) * 16.0588f + 0.5f), 0, 4095);
  if (duty < 8) duty = 0;
  int dF = 0, dR = 0;
  if (s > 0.05f) {
    dF = duty; dR = 0;
  } else if (s < -0.05f) {
    dF = 0; dR = duty;
  }
  if (lastDutyF[chF] != dF) {
    pwmWrite(rpwmPin, chF, dF);
    lastDutyF[chF] = dF;
  }
  if (lastDutyR[chR] != dR) {
    pwmWrite(lpwmPin, chR, dR);
    lastDutyR[chR] = dR;
  }
}

void writeAllDrives(float s0, float s1, float s2, float s3) {
  writeSingleDrive(DRV1_RPWM, DRV1_LPWM, CH_DRV1_F, CH_DRV1_R, s0, INV_DRV1);
  writeSingleDrive(DRV2_RPWM, DRV2_LPWM, CH_DRV2_F, CH_DRV2_R, s1, INV_DRV2);
  writeSingleDrive(DRV3_RPWM, DRV3_LPWM, CH_DRV3_F, CH_DRV3_R, s2, INV_DRV3);
  writeSingleDrive(DRV4_RPWM, DRV4_LPWM, CH_DRV4_F, CH_DRV4_R, s3, INV_DRV4);
}

// ============================================================
//  7. TÍNH TOÁN VẬN TỐC & 5 TẦNG LỌC ENCODER
// ============================================================
inline float median3(float a, float b, float c) {
  if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
  if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
  return c;
}

float calculateMovingAverage(float buffer[], int size) {
  float sum = 0.0f;
  for (int i = 0; i < size; i++) sum += buffer[i];
  return sum / size;
}

void calculateSpeed() {
  unsigned long now = millis();
  if (now - lastSpeedCalcTime < SPEED_CALC_INTERVAL_MS) return;
  float dt = (now - lastSpeedCalcTime) / 1000.0f;
  lastSpeedCalcTime = now;
  unsigned long now_us = micros();

  for (int i = 0; i < 4; i++) {
    noInterrupts();
    long curCount = enc[i].count;
    unsigned long lastEdgeUs = enc[i].lastEdgeTime;
    interrupts();

    long pulses = curCount - enc[i].lastSpeedCount;
    long pulses2 = curCount - enc[i].prevSpeedCount2;
    unsigned long edgeDeltaUs1 = lastEdgeUs - enc[i].prevCalcEdgeUs;
    unsigned long edgeDeltaUs2 = lastEdgeUs - enc[i].prevCalcEdgeUs2;

    enc[i].prevSpeedCount2 = enc[i].lastSpeedCount;
    enc[i].lastSpeedCount  = curCount;
    enc[i].prevCalcEdgeUs2 = enc[i].prevCalcEdgeUs;
    if (pulses != 0) {
      enc[i].prevCalcEdgeUs = lastEdgeUs;
    }

    // Tầng 1: Tấm khiên giới hạn vật lý (Tránh cắt cụt xung khi chu kỳ dt dao động)
    if (labs(pulses) > 600) {
      pulses = (pulses > 0) ? 600 : -600;
    }

    // Tầng 2: Chống trôi ảo khi dừng (Deadband Zero-Motion Suppression)
    if (!isMoving && wpid[i].targetRPM <= 0.1f && labs(pulses) <= 2) {
      pulses = 0;
    }

    // Tính độ lớn RPM bằng Phương pháp MT Cửa sổ 100ms (8-10 xung) + Khởi tạo tức thời bậc 1:
    // - Triệt tiêu hoàn toàn sai số bất đối xứng pha A/B và sai số lượng tử 4-5 xung.
    // - Chỉ dùng duy nhất bộ lọc bậc 1 (EMA alpha=0.70) có khởi tạo trực tiếp ở xung đầu tiên -> Độ trễ = 0, không thể dao động!
    float rawRPM = 0.0f;
    if (pulses != 0) {
      float dt_2 = edgeDeltaUs2 * 1e-6f;
      float dt_1 = edgeDeltaUs1 * 1e-6f;
      if (labs(pulses2) >= 6 && dt_2 >= dt * 1.35f && dt_2 <= dt * 2.75f) {
        rawRPM = (labs(pulses2) * 60.0f) / (ENCODER_CPR * dt_2);
      } else if (dt_1 >= dt * 0.55f && dt_1 <= dt * 1.80f) {
        rawRPM = (labs(pulses) * 60.0f) / (ENCODER_CPR * dt_1);
      } else {
        rawRPM = (labs(pulses) * 60.0f) / (ENCODER_CPR * dt);
      }
    } else {
      unsigned long idleUs = now_us - lastEdgeUs;
      if (!isMoving || idleUs > 120000UL) {
        rawRPM = 0.0f;
      } else {
        rawRPM = enc[i].prevFilteredRpm * 0.85f;
      }
    }

    // Tầng 3: Giới hạn vật lý cho động cơ 775 hộp số 220 RPM
    if (rawRPM > MAX_PLAUSIBLE_RPM) {
      rawRPM = MAX_PLAUSIBLE_RPM;
    }

    // Tầng 4: Bộ lọc bậc 1 duy nhất có Pre-Seed tức thời khi bắt đầu lăn bánh (Zero Phase Lag)
    if (rawRPM <= 0.05f) {
      enc[i].rpm = 0.0f;
      enc[i].prevFilteredRpm = 0.0f;
    } else if (enc[i].prevFilteredRpm < 1.2f) {
      enc[i].rpm = rawRPM;
      enc[i].prevFilteredRpm = rawRPM;
    } else {
      enc[i].rpm = 0.70f * rawRPM + 0.30f * enc[i].prevFilteredRpm;
      enc[i].prevFilteredRpm = enc[i].rpm;
    }

    enc[i].speed_ms   = enc[i].rpm * WHEEL_CIRCUMFERENCE / 60.0f;
    enc[i].distance_m += (labs(pulses) * WHEEL_CIRCUMFERENCE) / ENCODER_CPR;
  }

#if WHEEL3_ENCODER_FAULT
  enc[3].rpm        = enc[2].rpm;
  enc[3].speed_ms   = enc[2].speed_ms;
  enc[3].distance_m = enc[2].distance_m;
  enc[3].count      = enc[2].count;
#endif

  // 1. GÓI RAW COUNT + TIMESTAMP (US) GỬI LÊN RASPBERRY PI 5 (ROS 2 JAZZY)
  // Định dạng chuẩn: RAW <sequence> <timestamp_us> <FL> <FR> <RL> <RR>
  // Thứ tự bánh: FL=enc[0], FR=enc[2], RL=enc[1], RR=enc[3]
  static uint32_t raw_seq = 0;
  raw_seq++;
  Serial.printf("RAW %lu %lu %ld %ld %ld %ld\n", raw_seq, now_us, enc[0].count, enc[2].count, enc[1].count, enc[3].count);

  // 2. Gói Odometry phản hồi (20 Hz)
  float v_left  = (enc[0].speed_ms + enc[1].speed_ms) / 2.0f;
  float v_right = (enc[2].speed_ms + enc[3].speed_ms) / 2.0f;

  if (currentDirection == "ROS") {
    v_left  = (rosTargetRpmSigned[0] < 0) ? -fabsf(v_left) : fabsf(v_left);
    v_right = (rosTargetRpmSigned[2] < 0) ? -fabsf(v_right) : fabsf(v_right);
  } else {
    if (slew[0].target < 0) v_left  = -fabsf(v_left);
    if (slew[2].target < 0) v_right = -fabsf(v_right);
  }

  if (!isMoving || (fabsf(enc[0].rpm) < 1.0f && fabsf(enc[1].rpm) < 1.0f)) v_left = 0.0f;
  if (!isMoving || (fabsf(enc[2].rpm) < 1.0f && fabsf(enc[3].rpm) < 1.0f)) v_right = 0.0f;

  // Giao thức ODOM chuẩn ROS 2 (m/s) để tương thích ngược
  Serial.printf("ODOM %.3f %.3f\n", v_left, v_right);

  // Giao thức ENC 4 bánh (ticks & dt_ms)
  unsigned int dt_ms = (unsigned int)(dt * 1000.0f + 0.5f);
  Serial.printf("ENC %ld %ld %ld %ld %u\n", enc[0].count, enc[1].count, enc[2].count, enc[3].count, dt_ms);

  updateWheelHealth();
  updatePID(dt);
  Serial.printf("#PWM %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f\n",
                slew[0].current, slew[1].current, slew[2].current, slew[3].current,
                enc[0].rpm, enc[1].rpm, enc[2].rpm, enc[3].rpm);
}

void updateWheelHealth() {
  if (!isMoving) {
    for (int i = 0; i < 4; i++) wHealth[i] = {false, 0};
    return;
  }
  unsigned long now = millis();

  for (int i = 0; i < 4; i++) {
#if WHEEL3_ENCODER_FAULT
    if (i == 3) { wHealth[3] = wHealth[2]; continue; }
#endif
    float rpm = fabsf(enc[i].rpm);
    float pwm = fabsf(slew[i].target);

    // Phát hiện kẹt cứng cơ khí / hỏng encoder
    if (pwm > STALL_PWM_THRESHOLD && rpm < STALL_RPM_THRESHOLD) {
      if (wHealth[i].stallStartTime == 0) wHealth[i].stallStartTime = now;
      if (now - wHealth[i].stallStartTime > STALL_DETECT_MS && !wHealth[i].isStalled) {
        wHealth[i].isStalled = true;
        Serial.printf("[HEALTH] CANH BAO: Banh %d bi KHOA/HONG!\n", i + 1);
      }
    } else {
      wHealth[i].stallStartTime = 0;
      wHealth[i].isStalled = false;
    }
  }
}

// ============================================================
//  8. THUẬT TOÁN PID 12-BIT FLOAT + ĐỒNG BỘ 4 BÁNH SIÊU MƯỢT
// ============================================================
void updatePID(float dt) {
  if (!pidGlobalEnabled || !isMoving) return;

  float actualRPM[4];
  for (int j = 0; j < 4; j++) actualRPM[j] = fabsf(enc[j].rpm);

  // Phát hiện trượt bánh (Traction Control) theo cặp bánh cùng vế: Trái (0,1), Phải (2,3)
  static unsigned long slipStart[4] = {0, 0, 0, 0};
  bool slipping[4] = {false, false, false, false};
  unsigned long nowMs = millis();
  for (int j = 0; j < 4; j++) {
    int partner = (j == 0) ? 1 : (j == 1) ? 0 : (j == 2) ? 3 : 2;
    float tgt = wpid[j].targetRPM;
    bool over_target  = (actualRPM[j] > 25.0f) && (actualRPM[j] > tgt * SLIP_TARGET_RATIO + 2.0f);
    bool over_partner = actualRPM[j] > actualRPM[partner] * SLIP_PARTNER_RATIO + 5.0f;
    if (tgt > 1.0f && over_target && over_partner) {
      if (slipStart[j] == 0) slipStart[j] = nowMs;
      slipping[j] = (nowMs - slipStart[j]) >= SLIP_CONFIRM_MS;
    } else {
      slipStart[j] = 0;
    }
  }

  for (int i = 0; i < 4; i++) {
#if WHEEL3_ENCODER_FAULT
    if (i == 3) {
      wpid[3].pwmOutput     = wpid[2].pwmOutput;
      wpid[3].prevPwmOutput = wpid[2].prevPwmOutput;
      slew[3].target        = slew[2].target;
      continue;
    }
#endif
    if (!wpid[i].enabled) continue;

    float rpm_act = actualRPM[i];
    float target  = wpid[i].targetRPM;
    bool  isLeft  = (i == 0 || i == 1);
    int   partner = (i == 0) ? 1 : (i == 1) ? 0 : (i == 2) ? 3 : 2;
    bool  isPivot = (rosTargetRpmSigned[0] * rosTargetRpmSigned[2] < -0.5f);

    if (target <= 0.1f) {
      wpid[i].pwmOutput = 0.0f;
      wpid[i].integral = 0.0f;
      wpid[i].lastError = 0.0f;
      wpid[i].filteredDeriv = 0.0f;
      slew[i].target = 0.0f;
      continue;
    }

    // Bảo vệ khi 1 bánh bị hỏng encoder:
    // Nếu bánh i báo 0 RPM nhưng bánh cùng vế (partner) đang quay tốt (> 2.0 RPM) và đã cấp điện áp > 30 PWM:
    // -> Đồng bộ vận tốc thực tế theo bánh cùng vế để bánh hỏng encoder luôn quay đều cùng đồng đội, không bị khóa cứng hay vọt tốc!
    bool partnerHealthy = (actualRPM[partner] > 2.0f);
    bool encoderFaultSuspected = (rpm_act < 0.4f && partnerHealthy && wpid[i].prevPwmOutput > 30.0f);
    if (encoderFaultSuspected) {
      rpm_act = actualRPM[partner];
    }

    // 1. Feedforward số thực 12-bit chuẩn xác theo đường đặc tính 775 24V:
    // - Khi chạy thẳng bám luống (!isPivot): ff_pwm = 10.6 + (target/220)*65 (12.61 PWM tại 6.8 RPM), giữ 100% độ êm mượt.
    // - Khi xoay tại chỗ 4 bánh ngược chiều (isPivot): cấp ngay ff_pwm = 38.0 + (target/220)*95 (~44 PWM) và min_floor = 32.0f
    //   để thắng ma sát trượt ngang trên đất cỏ gồ ghề, giúp cả 4 bánh bứt phá đồng thời không bánh nào bị lì/khóa!
    float ff_pwm    = isPivot ? (38.0f + (target / 220.0f) * 95.0f) : (10.6f + (target / 220.0f) * 65.0f);
    float min_floor = isPivot ? 32.0f : 9.5f;

    // Trong 180ms đầu tiên khi vừa khởi động tiến thẳng từ trạng thái đứng yên (!isPivot):
    // Giữ nguyên điện áp Feedforward phẳng tuyệt đối để bánh xe gia tốc tự nhiên theo quán tính lên 6.8 RPM mà không bị tích phân nhồi vọt lố!
    if (!isPivot && (nowMs - startMoveTime < 180UL)) {
      wpid[i].integral      = 0.0f;
      wpid[i].lastError     = 0.0f;
      wpid[i].filteredDeriv = 0.0f;
      wpid[i].pwmOutput     = ff_pwm;
      wpid[i].prevPwmOutput = ff_pwm;
      float sign = (currentDirection == "ROS") ? ((rosTargetRpmSigned[i] >= 0.0f) ? 1.0f : -1.0f) : 1.0f;
      slew[i].target = sign * wpid[i].pwmOutput;
      continue;
    }

    // 2. Sai số bám tốc độ mục tiêu độc lập cho từng bánh
    float track_error = target - rpm_act;
    if (fabsf(track_error) < PID_ERROR_DEADBAND) {
      track_error = 0.0f;
    }

    // Phát hiện bánh quay trơn (trượt bánh khi giảm tải góc chéo lúc quay đầu trên đất gồ ghề):
    bool pivotWheelSpin = isPivot && (rpm_act > target * 1.25f) && (rpm_act > actualRPM[partner] * 1.25f + 2.0f);

    // 3. Khâu tích phân (Integral) số thực siêu mượt:
    // - Khi đang xoay tại chỗ (isPivot): nhân 3.0x tốc độ đùn tích phân lên tới +210.0 để thắng lực cản ngang của cỏ dày.
    // - Khi thiếu tốc mạnh lúc đi thẳng (gặp mô đất, rpm_act < 50% target): nhân 2.2x tốc độ đùn tích phân.
    // - Khi bánh bị trượt quay trơn (slipping / pivotWheelSpin): hạ tích phân êm dịu để bánh bám đất trở lại.
    if (!slipping[i] && !pivotWheelSpin) {
      float int_gain = 1.0f;
      if (isPivot && track_error > 0.8f) {
        int_gain = 3.0f;
      } else if (rpm_act < target * 0.50f && track_error > 1.5f) {
        int_gain = 2.2f;
      }
      wpid[i].integral = constrain(wpid[i].integral + track_error * int_gain * dt, -2.5f, 210.0f);
    } else {
      wpid[i].integral *= 0.88f;
    }

    // 4. Khâu vi phân (Derivative)
    float rawDeriv = (track_error - wpid[i].lastError) / dt;
    wpid[i].filteredDeriv = DERIVATIVE_FILTER * rawDeriv + (1.0f - DERIVATIVE_FILTER) * wpid[i].filteredDeriv;
    wpid[i].lastError = track_error;

    // 5. Tính toán PID số thực (Tăng cường Kp & Ki khi xoay tại chỗ để 4 bánh quay cực khỏe, không bị khóa bánh)
    float kp_eff = isPivot ? 1.50f : wpid[i].kp;
    float ki_eff = isPivot ? 2.40f : wpid[i].ki;
    float pid_corr = (kp_eff * track_error) + (ki_eff * wpid[i].integral) + (wpid[i].kd * wpid[i].filteredDeriv);

    // 6. Tổng hợp PWM số thực (0.0 - 255.0 -> ánh xạ sang 0 - 4095 ở writeSingleDrive)
    float desired = constrain(ff_pwm + pid_corr, 0.0f, 235.0f);

    if (target > 1.0f && desired < min_floor) {
      desired = min_floor;
    }

    if (slipping[i] || pivotWheelSpin) {
      desired = fminf(desired, wpid[i].prevPwmOutput * SLIP_PWM_CUT);
    }

    // Đồng bộ công suất cặp bánh cùng vế khi 1 bánh nghi hỏng encoder
    if (encoderFaultSuspected && wpid[partner].pwmOutput > min_floor) {
      desired = 0.35f * desired + 0.65f * wpid[partner].pwmOutput;
    }

    // Giới hạn biến thiên PWM vi sai:
    // - Khi đang xoay tại chỗ (isPivot): cho phép tăng +6.0 / giảm -4.0 PWM/50ms (+120 PWM/s) để cấp lực xoay tức thời thắng cỏ dày.
    // - Khi bánh đang bị cản mạnh bởi mô đất (rpm_act < 45% target): cho phép tăng +2.2 PWM/50ms (+44 PWM/s) để leo vượt mô đất.
    // - Khi bánh đã lăn ổn định: khóa bước biến thiên ở +0.30 / -0.25 PWM mỗi 50ms (< 0.1% điện áp) -> Triệt tiêu 100% hiện tượng gật bánh!
    float stepUpLim, stepDownLim;
    if (isPivot) {
      stepUpLim   = 6.0f;
      stepDownLim = 4.0f;
    } else if (rpm_act < target * 0.45f && track_error > 1.5f) {
      stepUpLim   = 2.20f;
      stepDownLim = 0.25f;
    } else {
      stepUpLim   = 0.30f;
      stepDownLim = 0.25f;
    }
    float maxChange   = (desired > wpid[i].prevPwmOutput) ? stepUpLim : stepDownLim;
    float delta       = constrain(desired - wpid[i].prevPwmOutput, -maxChange, maxChange);
    wpid[i].pwmOutput     = constrain(wpid[i].prevPwmOutput + delta, 0.0f, 235.0f);
    wpid[i].prevPwmOutput = wpid[i].pwmOutput;

    // Nếu 1 bánh báo kẹt/hỏng encoder quá 4.0s: nếu bánh cùng vế vẫn quay thì đồng bộ theo bánh cùng vế, ngược lại giới hạn an toàn 160 PWM
    if (wHealth[i].isStalled) {
      if (partnerHealthy && wpid[partner].pwmOutput > 20.0f) {
        wpid[i].pwmOutput     = wpid[partner].pwmOutput;
        wpid[i].prevPwmOutput = wpid[partner].pwmOutput;
      } else {
        wpid[i].pwmOutput = fminf(wpid[i].pwmOutput, 160.0f);
        wpid[i].integral  = constrain(wpid[i].integral, -5.0f, 120.0f);
      }
    }

    // Xác định chiều quay
    float sign = 0.0f;
    if (currentDirection == "ROS")            sign = (rosTargetRpmSigned[i] >= 0.0f) ? 1.0f : -1.0f;
    else if (currentDirection == "FORWARD")   sign = 1.0f;
    else if (currentDirection == "BACKWARD")  sign = -1.0f;
    else if (currentDirection == "LEFT")      sign = isLeft ? -1.0f : 1.0f;
    else if (currentDirection == "RIGHT")     sign = isLeft ? 1.0f : -1.0f;

    slew[i].target = sign * wpid[i].pwmOutput;
  }
}

// ============================================================
//  9. SLEW RATE LIMITER — NỘI SUY XUNG 12-BIT FLOAT (40 Hz)
// ============================================================
void updateSpeedRamp() {
  unsigned long now = millis();
  if (now - lastRampTime < RAMP_INTERVAL_MS) return;
  lastRampTime = now;

  // Phát hiện chế độ xoay tại chỗ (pivot): 2 vế bánh quay ngược chiều
  bool isPivotSlew = (slew[0].target * slew[2].target < -0.5f);

  for (int i = 0; i < 4; i++) {
    float &cur  = slew[i].current;
    float  tgt  = slew[i].target;
    float &step = slew[i].step;

    if (fabsf(tgt) < 0.05f) {
      step = 3.0f;  // Hạ tốc êm ái khi dừng
    } else if (isPivotSlew) {
      step = 4.0f;  // Pivot turn: nội suy nhanh để thắng ma sát trượt ngang
    } else if (fabsf(tgt) > fabsf(cur)) {
      step = 1.5f;  // Nội suy 40Hz siêu mượt từ đầu ra PID 20Hz
    } else {
      step = 1.0f;  // Giảm tốc êm như lụa
    }

    if (fabsf(tgt - cur) <= step) cur = tgt;
    else if (cur < tgt)           cur += step;
    else                          cur -= step;
  }

  writeSingleDrive(DRV1_RPWM, DRV1_LPWM, CH_DRV1_F, CH_DRV1_R, slew[0].current, INV_DRV1);
  writeSingleDrive(DRV2_RPWM, DRV2_LPWM, CH_DRV2_F, CH_DRV2_R, slew[1].current, INV_DRV2);
  writeSingleDrive(DRV3_RPWM, DRV3_LPWM, CH_DRV3_F, CH_DRV3_R, slew[2].current, INV_DRV3);
  writeSingleDrive(DRV4_RPWM, DRV4_LPWM, CH_DRV4_F, CH_DRV4_R, slew[3].current, INV_DRV4);
}

// ============================================================
//  10. DỪNG XE & CẬP NHẬT TỐC ĐỘ XE
// ============================================================
void stopMotor(bool emergency) {
  currentDirection = "STOP";

  for (int i = 0; i < 4; i++) {
    slew[i].target = 0;
    wpid[i].targetRPM = 0.0f;
    wpid[i].integral = 0;
    wpid[i].lastError = 0;
    wpid[i].filteredDeriv = 0;
    wpid[i].pwmOutput = 0;
    wpid[i].prevPwmOutput = 0;
    rosTargetRpmSigned[i] = 0.0f;
  }

  if (emergency) {
    for (int i = 0; i < 4; i++) slew[i].current = 0.0f;
    writeAllDrives(0, 0, 0, 0);
    isMoving = false;
    Serial.println("# [ESTOP] DUNG KHAN CAP - NGAT DIEN TUC THI!");
  }
}

void stopMotor() {
  stopMotor(false);
}

void setVehicleSpeed(int spd) {
  currentSpeed = constrain(spd, 0, 255);
  if (currentSpeed == 0) {
    stopMotor(false);
    return;
  }
  globalTargetRPM = (currentSpeed / 255.0f) * 220.0f;
  for (int i = 0; i < 4; i++) {
    wpid[i].targetRPM = globalTargetRPM;
  }
}

// ============================================================
//  11. XỬ LÝ LỆNH SERIAL (LỆNH IP & LỆNH ĐIỀU KHIỂN)
// ============================================================
void handleCommand(String command) {
  command.trim();
  if (command.length() == 0) return;

  // Lệnh kiểm tra IP kết nối từ Serial Monitor: "IP"
  if (command.equalsIgnoreCase("IP")) {
    Serial.println();
    Serial.println("=========================================");
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print  ("  >> IP XE THUC TE : ");
      Serial.println(WiFi.localIP());
      Serial.printf ("  >> WiFi SSID     : %s (RSSI: %d dBm)\n", WiFi.SSID().c_str(), WiFi.RSSI());
    } else {
      Serial.println("  [CHUA CO IP] Xe chua ket noi WiFi!");
      Serial.printf ("  SSID dang cau hinh: '%s'\n", ssid);
    }
    Serial.println("=========================================\n");
    return;
  }

  // Lệnh vận tốc ROS 2: "V <FL> <RL> <FR> <RR>" hoặc "V <L> <R>" (RPM)
  if (command.startsWith("V ") || command.startsWith("v ") || command.startsWith("V\t") || command.startsWith("v\t")) {
    float r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int parsed = sscanf(command.c_str() + 2, "%f %f %f %f", &r[0], &r[1], &r[2], &r[3]);
    if (parsed == 2) {
      r[3] = r[1]; // Bánh phải sau = Bánh phải trước
      r[2] = r[1];
      r[1] = r[0]; // Bánh trái sau = Bánh trái trước
      parsed = 4;
    }
    if (parsed == 4) {
      lastRosCmdTime = millis();
      if (fabsf(r[0]) < 0.1f && fabsf(r[1]) < 0.1f && fabsf(r[2]) < 0.1f && fabsf(r[3]) < 0.1f) {
        stopMotor(false);
      } else {
        bool wasStopped = !isMoving;
        bool isPivotCmd = (r[0] * r[2] < -0.5f);
        if (wasStopped && !isPivotCmd) {
          startMoveTime = millis();
        }
        isMoving = true;
        currentDirection = "ROS";
        for (int i = 0; i < 4; i++) {
          float tgt_abs = fabsf(r[i]);
          float init_ff = isPivotCmd ? (38.0f + (tgt_abs / 220.0f) * 95.0f) : (10.6f + (tgt_abs / 220.0f) * 65.0f);
          float sgn     = (r[i] >= 0.0f) ? 1.0f : -1.0f;
          if (wasStopped || (rosTargetRpmSigned[i] * r[i] < -0.01f)) {
            if (!isPivotCmd && (rosTargetRpmSigned[i] * r[i] < -0.01f)) startMoveTime = millis();
            wpid[i].integral      = 0.0f;
            wpid[i].lastError     = 0.0f;
            wpid[i].filteredDeriv = 0.0f;
            wpid[i].pwmOutput     = init_ff;
            wpid[i].prevPwmOutput = init_ff;
            slew[i].current       = sgn * init_ff;
            slew[i].target        = sgn * init_ff;
            enc[i].prevFilteredRpm = 0.0f;
          }
          rosTargetRpmSigned[i] = r[i];
          wpid[i].targetRPM = tgt_abs;
          wpid[i].enabled = pidGlobalEnabled;
        }
      }
      return;
    }
  }

  // Lệnh Reset Odometry
  if (command.equalsIgnoreCase("RESET_ODOM") || command.equalsIgnoreCase("ZERO") || command.equalsIgnoreCase("RESET")) {
    for (int i = 0; i < 4; i++) {
      enc[i].count = 0;
      enc[i].distance_m = 0.0f;
      enc[i].lastSpeedCount = 0;
    }
    Serial.println("# [ODOM] Da reset toan bo xung va quang duong ve 0.00m");
    return;
  }

  // Lệnh PID: Luôn giữ PID bật cố định để đảm bảo đồng tốc 4 bánh chính xác
  if (command.equalsIgnoreCase("PID 1") || command.equalsIgnoreCase("PID ON")) {
    pidGlobalEnabled = true;
    for (int i = 0; i < 4; i++) wpid[i].enabled = true;
    Serial.println("# [MODE] PID: ON (Luon bat)");
    return;
  }
  if (command.equalsIgnoreCase("PID 0") || command.equalsIgnoreCase("PID OFF")) {
    // Luôn giữ PID bật theo yêu cầu hệ thống, không cho tắt để bảo vệ cân bằng 4 bánh
    pidGlobalEnabled = true;
    for (int i = 0; i < 4; i++) wpid[i].enabled = true;
    Serial.println("# [MODE] PID: LUON BAT (Locked ON - Khong the tat de giu dong toc)");
    return;
  }

  // Lệnh Dừng xe
  if (command.equalsIgnoreCase("STOP") || command.equalsIgnoreCase("X") || command == " ") {
    stopMotor(false);
    return;
  }
  if (command.equalsIgnoreCase("ESTOP") || command.equalsIgnoreCase("EMERGENCY_STOP")) {
    stopMotor(true);
    return;
  }

  // Lệnh chỉnh tốc độ: "SPEED <val>"
  if (command.startsWith("SPEED ") || command.startsWith("speed ")) {
    int spd = command.substring(6).toInt();
    if (spd > 0) {
      setVehicleSpeed(spd);
      Serial.printf("# [SPEED] Target RPM = %.1f (PWM %d)\n", globalTargetRPM, currentSpeed);
    }
    return;
  }

  // Lệnh test thủ công từ Serial Monitor: F (Tiến), B (Lùi), L (Trái), R (Phải)
  command.toUpperCase();
  if (command == "F" || command == "FORWARD") {
    currentDirection = "FORWARD";
    isMoving = true;
    for (int i = 0; i < 4; i++) wpid[i].targetRPM = globalTargetRPM;
    return;
  } else if (command == "B" || command == "BACKWARD") {
    currentDirection = "BACKWARD";
    isMoving = true;
    for (int i = 0; i < 4; i++) wpid[i].targetRPM = globalTargetRPM;
    return;
  } else if (command == "L" || command == "LEFT") {
    currentDirection = "LEFT";
    isMoving = true;
    for (int i = 0; i < 4; i++) wpid[i].targetRPM = globalTargetRPM;
    return;
  } else if (command == "R" || command == "RIGHT") {
    currentDirection = "RIGHT";
    isMoving = true;
    for (int i = 0; i < 4; i++) wpid[i].targetRPM = globalTargetRPM;
    return;
  }
}

// ============================================================
//  12. SETUP CẤU HÌNH PHẦN CỨNG & KẾT NỐI WIFI LẤY IP
// ============================================================
void setup() {
  lockAllDriverPins();

  Serial.begin(115200);
  Serial.setTimeout(10);

  pwmSetup(DRV1_RPWM, CH_DRV1_F); pwmSetup(DRV1_LPWM, CH_DRV1_R);
  pwmSetup(DRV2_RPWM, CH_DRV2_F); pwmSetup(DRV2_LPWM, CH_DRV2_R);
  pwmSetup(DRV3_RPWM, CH_DRV3_F); pwmSetup(DRV3_LPWM, CH_DRV3_R);
  pwmSetup(DRV4_RPWM, CH_DRV4_F); pwmSetup(DRV4_LPWM, CH_DRV4_R);
  writeAllDrives(0, 0, 0, 0);

  // Khởi tạo trạng thái ban đầu của 4 Encoder và gắn ngắt Quadrature X4 trên cả 2 pha
  for (int i = 0; i < 4; i++) {
    pinMode(enc[i].pinA, INPUT_PULLUP);
    pinMode(enc[i].pinB, INPUT_PULLUP);
    uint8_t a = digitalRead(enc[i].pinA);
    uint8_t b = digitalRead(enc[i].pinB);
    enc[i].state = (a << 1) | b;
    enc[i].lastEdgeTime = micros();
    wpid[i].enabled = true;
    wpid[i].targetRPM = globalTargetRPM;
  }

  attachInterrupt(digitalPinToInterrupt(enc[0].pinA), isr_enc0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(enc[0].pinB), isr_enc0, CHANGE);

  attachInterrupt(digitalPinToInterrupt(enc[1].pinA), isr_enc1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(enc[1].pinB), isr_enc1, CHANGE);

  attachInterrupt(digitalPinToInterrupt(enc[2].pinA), isr_enc2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(enc[2].pinB), isr_enc2, CHANGE);

  attachInterrupt(digitalPinToInterrupt(enc[3].pinA), isr_enc3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(enc[3].pinB), isr_enc3, CHANGE);

  Serial.println("\n# =========================================================");
  Serial.println("#   ESP32 4-WHEEL DIFFERENTIAL DRIVE CONTROLLER FOR ROS 2  ");
  Serial.printf ("#   Encoder: %d PPR | Quadrature: X4 -> %d CPR              \n", ENCODER_PPR, ENCODER_CPR);
  Serial.printf ("#   Wheel: %.0fmm | Loop: 20Hz RAW & ODOM                  \n", WHEEL_DIAMETER_M * 1000);
  Serial.println("# =========================================================");

  // Khởi tạo WiFi Station để nhận IP
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("\n# Dang ket noi WiFi '%s' de lay IP...", ssid);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 15) {
    delay(400);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("# =========================================================");
    Serial.println("#  WIFI KET NOI THANH CONG!");
    Serial.print  ("#  >> IP XE THUC TE : ");
    Serial.println(WiFi.localIP());
    Serial.printf ("  >> WiFi SSID     : %s (RSSI: %d dBm)\n", WiFi.SSID().c_str(), WiFi.RSSI());
    Serial.println("#  (Go lenh 'IP' tren Serial Monitor de xem lai bat ky luc nao)");
    Serial.println("# =========================================================");
  } else {
    Serial.println("# [THONG BAO] Chưa ket noi duoc WiFi. Van chay Serial ROS 2 binh thuong.");
  }

  Serial.println("# Ready! Dang truyen RAW count va nhan lenh ROS 2 qua cong Serial...\n");
}

// ============================================================
//  13. VÒNG LẶP CHÍNH (LOOP)
// ============================================================
void loop() {
  // Đọc Serial non-blocking từ Raspberry Pi ROS 2
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialRxBuf.length() > 0) {
        handleCommand(serialRxBuf);
        serialRxBuf = "";
      }
    } else {
      if (serialRxBuf.length() < 64) serialRxBuf += c;
    }
  }

  // Khởi động mềm Slew Rate (40 Hz)
  updateSpeedRamp();

  // Dừng xe mượt mà khi lệnh là STOP
  if (currentDirection == "STOP" && isMoving) {
    bool allStopped = true;
    for (int i = 0; i < 4; i++) {
      if (fabsf(slew[i].current) > 2.0f || fabsf(enc[i].rpm) > 1.5f) {
        allStopped = false;
        break;
      }
    }
    if (allStopped) {
      writeAllDrives(0, 0, 0, 0);
      isMoving = false;
    }
  }

  // ROS Watchdog an toàn: Tự ngắt nếu quá 1.5s Raspberry Pi không gửi lệnh
  if (currentDirection == "ROS" && isMoving) {
    if (millis() - lastRosCmdTime > ROS_WATCHDOG_TIMEOUT_MS) {
      stopMotor(false);
      Serial.println("# [WATCHDOG] Pi mat ket noi qua 1500ms -> Ngat xe an toan!");
    }
  }

  // Chu kỳ tính toán tốc độ, lọc 5 tầng và gửi RAW / ODOM / ENC (20 Hz)
  calculateSpeed();

  delay(1);
}
