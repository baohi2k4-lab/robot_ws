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

// Giới hạn gia tốc & Vùng an toàn PWM
#define MIN_PWM                 0         // Cho phép toàn dải 0-255 PWM cho tốc độ cực chậm
#define MAX_PWM_CHANGE_UP       45        // Bước tăng PWM tối đa mỗi chu kỳ (tăng lực nhanh, dứt khoát đẩy tải nặng)
#define MAX_PWM_CHANGE_DOWN     25        // Bước giảm PWM tối đa (bảo vệ cơ cấu nhông hộp số)
#define RAMP_STEP_MAX           12.0f     // Bước ramp gia tốc tối đa
#define RAMP_STEP_MIN           4.0f      // Bước ramp khởi động dứt khoát để thắng tải nặng
#define RAMP_STEP_STOP_MAX      5.0f      // Bước giảm tốc êm ái khi dừng

// Lọc nhiễu Encoder & Ngưỡng vật lý
#define MIN_ENC_INTERVAL_US     15        // Chặn gai nhiễu megahertz tia lửa chổi than (<15us)
#define MAX_PLAUSIBLE_RPM       350.0f    // Ngưỡng RPM tối đa vật lý (loại bỏ đột biến)
#define MOVING_AVG_SIZE         2         // Kích thước mảng trượt ngắn (loại bỏ trễ pha 150-200ms)
#define DERIVATIVE_FILTER       0.6f      // Hệ số lọc thông thấp cho khâu vi phân PID
#define EMA_LOW_SPEED_ALPHA     0.70f     // Phản hồi tốc độ tức thời (<40 RPM), không gây trễ vòng lặp PID
#define EMA_HIGH_SPEED_ALPHA    0.80f     // EMA tốc độ cao (phản hồi nhanh)
#define PID_ERROR_DEADBAND      0.3f      // Vùng chết sai số PID chuẩn cho tốc độ chậm (0.06 m/s)

// Giám sát an toàn & Kẹt bánh (Stall Detection)
#define STALL_DETECT_MS         1200      // Cho phép 1.2s mô-men xoắn cực đại trước khi cảnh báo
#define STALL_PWM_THRESHOLD     180
#define STALL_RPM_THRESHOLD     4.0f

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

const int PWM_FREQ = 7000;
const int PWM_RES  = 8;      // 8-bit: 0 - 255
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
  long  lastSpeedCount;
  float rpm;
  float prevFilteredRpm;
  float speed_ms;
  float distance_m;
  float rpmBuffer[MOVING_AVG_SIZE];
  int   bufferIndex;
  float rawRpmHist[2];              // Bộ nhớ 2 chu kỳ cho Bộ lọc Trung vị 3 điểm (Median-3)
};

// Sơ đồ 4 Encoder: Trái trước (16,17), Trái sau (38,39), Phải trước (40,41), Phải sau (10,11)
EncoderData enc[4] = {
  {16, 17,  1, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[0]: FL (Trái trước)
  {38, 39,  1, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[1]: RL (Trái sau)
  {40, 41, -1, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}, // enc[2]: FR (Phải trước)
  {10, 11, -1, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {0}, 0, {0.0f, 0.0f}}  // enc[3]: RR (Phải sau)
};

struct WheelPID {
  float kp, ki, kd;
  float targetRPM;
  float integral;
  float lastError;
  float filteredDeriv;
  int   pwmOutput;
  int   prevPwmOutput;
  bool  enabled;
};

// Hệ số PID tối ưu bám tốc và vượt gờ đất gồ ghề cho 4 động cơ 775
#define PID_KP              0.950f  // Kp tăng lực đẩy nhanh khi gặp mô đất
#define PID_KI              0.800f  // Ki tích phân bù lực tức thì vượt ma sát đất gồ ghề
#define PID_KD              0.035f  // Kd nhỏ để hãm nhẹ
#define K_SYNC_CROSS_WHEEL  0.0f    // Khử hoàn toàn giằng co chéo bánh
#define K_LR_BALANCE        0.0f    // Để ROS tự chủ động điều phối góc quay Wz

WheelPID wpid[4] = {
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, true},
  {PID_KP, PID_KI, PID_KD, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, true}
};

// Bộ lọc Kalman 1D cho từng bánh xe (Tách nhiễu Gauss & triệt tiêu trễ pha)
struct Kalman1D {
  float x; // Ước lượng vận tốc RPM
  float p; // Sai số ước lượng
  float q; // Nhiễu quá trình mô hình
  float r; // Nhiễu đo lường encoder
  float k; // Hệ số tăng Kalman
};

Kalman1D kfRpm[4] = {
  {0.0f, 1.0f, 0.10f, 6.0f, 0.0f},
  {0.0f, 1.0f, 0.10f, 6.0f, 0.0f},
  {0.0f, 1.0f, 0.10f, 6.0f, 0.0f},
  {0.0f, 1.0f, 0.10f, 6.0f, 0.0f}
};

inline float updateKalman1D(Kalman1D &kf, float measurement) {
  kf.p = kf.p + kf.q;
  kf.k = kf.p / (kf.p + kf.r);
  kf.x = kf.x + kf.k * (measurement - kf.x);
  kf.p = (1.0f - kf.k) * kf.p;
  return kf.x;
}

struct SlewChannel {
  int   target;                     // Tốc độ mục tiêu (-255 đến +255)
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
int           currentSpeed            = 200;
bool          isMoving                = false;
bool          pidGlobalEnabled        = true;
float         globalTargetRPM         = 60.0f;
String        currentDirection        = "STOP";
float         rosTargetRpmSigned[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
unsigned long lastRosCmdTime          = 0;
String        serialRxBuf             = "";

unsigned long lastSpeedCalcTime       = 0;
unsigned long lastRampTime            = 0;

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

void writeSingleDrive(int rpwmPin, int lpwmPin, int chF, int chR, int speed, int inv) {
  speed = speed * inv;
  if (speed > 0) {
    pwmWrite(rpwmPin, chF, speed);
    pwmWrite(lpwmPin, chR, 0);
  } else if (speed < 0) {
    pwmWrite(rpwmPin, chF, 0);
    pwmWrite(lpwmPin, chR, -speed);
  } else {
    pwmWrite(rpwmPin, chF, 0);
    pwmWrite(lpwmPin, chR, 0);
  }
}

void writeAllDrives(int s0, int s1, int s2, int s3) {
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
    interrupts();

    long pulses = curCount - enc[i].lastSpeedCount;
    enc[i].lastSpeedCount = curCount;

    // Tầng 1: Tấm khiên giới hạn vật lý (X4 CPR = 800 -> 350 RPM ≈ 233 xung/50ms)
    if (labs(pulses) > 240) {
      pulses = (pulses > 0) ? 240 : -240;
    }

    // Tầng 2: Chống trôi ảo khi dừng (Deadband Zero-Motion Suppression)
    if (!isMoving && wpid[i].targetRPM <= 0.1f && labs(pulses) <= 2) {
      pulses = 0;
    }

    // Tính RPM thô dựa trên CPR = 800
    float rawRPM = (pulses * 60.0f) / (ENCODER_CPR * dt);

    // Tầng 3: Bộ lọc trung vị 3 điểm (Median-3 Filter)
    float medRPM = median3(rawRPM, enc[i].rawRpmHist[0], enc[i].rawRpmHist[1]);
    enc[i].rawRpmHist[1] = enc[i].rawRpmHist[0];
    enc[i].rawRpmHist[0] = rawRPM;

    // Tầng 4: Bộ lọc quán tính chống đột biến (Physical Inertia Outlier Filter)
    if (fabsf(medRPM) > MAX_PLAUSIBLE_RPM || (isMoving && fabsf(medRPM - enc[i].prevFilteredRpm) > 50.0f)) {
      medRPM = enc[i].prevFilteredRpm;
    }

    enc[i].rpmBuffer[enc[i].bufferIndex] = medRPM;
    enc[i].bufferIndex = (enc[i].bufferIndex + 1) % MOVING_AVG_SIZE;

    // Tầng 5a: Bộ lọc trung bình trượt + EMA thích ứng tốc độ
    float movingAvgRPM = calculateMovingAverage(enc[i].rpmBuffer, MOVING_AVG_SIZE);
    float alpha = (fabsf(movingAvgRPM) < 40.0f) ? EMA_LOW_SPEED_ALPHA : EMA_HIGH_SPEED_ALPHA;
    float emaRpm = alpha * movingAvgRPM + (1.0f - alpha) * enc[i].prevFilteredRpm;
    enc[i].prevFilteredRpm = emaRpm;

    // Tầng 5b: BỘ LỌC KALMAN 1D TỐI ƯU (Khử trễ pha, cung cấp vận tốc chuẩn cho PID)
    enc[i].rpm = updateKalman1D(kfRpm[i], emaRpm);

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
    float rpm = abs(enc[i].rpm);
    int   pwm = abs(slew[i].target);

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
//  8. THUẬT TOÁN PID + TORQUE BOOST + ĐỒNG BỘ 4 BÁNH
// ============================================================
void updatePID(float dt) {
  if (!pidGlobalEnabled || !isMoving) return;

  float actualRPM[4];
  for (int j = 0; j < 4; j++) actualRPM[j] = fabsf(enc[j].rpm);

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

    if (target <= 0.1f) {
      wpid[i].pwmOutput = 0;
      wpid[i].integral = 0;
      wpid[i].lastError = 0;
      wpid[i].filteredDeriv = 0;
      slew[i].target = 0;
      continue;
    }

    // 1. Feedforward có bù ma sát tĩnh thích ứng (Adaptive Friction Compensation)
    // - Khi xoay tại chỗ (Pivot Turn): sàn 45 PWM (min_floor 36) đủ mô-men xoay đầm chắc
    // - Khi bò chậm (target <= 8.5 RPM ~ 0.06 - 0.08 m/s): hạ sàn về 22-30 PWM để chạy đúng 0.06 m/s
    // - Khi chạy nhanh: nâng sàn tuyến tính
    bool isPivotTurn = (rosTargetRpmSigned[0] * rosTargetRpmSigned[2] < -0.01f);
    float base_floor;
    int   min_floor;
    if (isPivotTurn) {
      base_floor = 45.0f;
      min_floor  = 36;
    } else {
      if (target <= 8.5f) {
        // Dải bò chậm 0.06 m/s (~5.7 RPM)
        base_floor = 22.0f + (target / 8.5f) * 8.0f; // 22.0 -> 30.0 PWM
        min_floor  = 20;
      } else {
        base_floor = 30.0f + ((target - 8.5f) / 211.5f) * 25.0f; // 30 -> 55 PWM
        min_floor  = 26;
      }
    }
    float ff_pwm = base_floor + (target / 220.0f) * (255.0f - base_floor);

    // 2. Sai số bám tốc độ mục tiêu độc lập cho từng bánh
    float track_error = target - rpm_act;
    if (fabsf(track_error) < PID_ERROR_DEADBAND) {
      track_error = 0.0f;
    }

    // 3. Khâu tích phân (Integral) có Anti-Windup & Tự động bơm lực khi gặp đất gồ ghề
    // Khi chạy chậm (target <= 8.5 RPM), sai số tuyệt đối nhỏ nhưng xe cần mô-men lớn để trèo gờ đất:
    float eff_error = track_error;
    if (target <= 8.5f && track_error > 0.0f) {
      // Bánh đang quay chậm hơn lệnh (bị cản bởi mô đất/gờ rãnh) -> khuếch đại sai số để tích phân bơm lực nhanh
      eff_error *= 2.5f;
    }
    wpid[i].integral = constrain(wpid[i].integral + eff_error * dt, -140.0f, 140.0f);

    // 4. Khâu vi phân (Derivative) có lọc nhiễu tần số cao
    float rawDeriv = (track_error - wpid[i].lastError) / dt;
    wpid[i].filteredDeriv = DERIVATIVE_FILTER * rawDeriv + (1.0f - DERIVATIVE_FILTER) * wpid[i].filteredDeriv;
    wpid[i].lastError = track_error;

    // 5. Tính toán PID cơ sở (sử dụng eff_error để bơm lực tức thì khi kẹt dốc)
    float pid_corr = (wpid[i].kp * eff_error) + (wpid[i].ki * wpid[i].integral) + (wpid[i].kd * wpid[i].filteredDeriv);

    // 6. Tổng hợp PWM điều khiển hoàn chỉnh (0-255)
    int desired = constrain((int)(ff_pwm + pid_corr), 0, 255);

    // Sàn PWM tối thiểu để động cơ 775 duy trì lăn bánh khỏe khoắn trên đất gồ ghề
    if (target > 1.0f && desired < min_floor) {
      desired = min_floor;
    }

    // Giới hạn biến thiên PWM bất đối xứng (chống giật sốc cơ khí)
    int maxChange = (desired > wpid[i].prevPwmOutput) ? MAX_PWM_CHANGE_UP : MAX_PWM_CHANGE_DOWN;
    int delta   = constrain(desired - wpid[i].prevPwmOutput, -maxChange, maxChange);
    wpid[i].pwmOutput     = constrain(wpid[i].prevPwmOutput + delta, 0, 255);
    wpid[i].prevPwmOutput = wpid[i].pwmOutput;

    // Giảm tải cho bánh bị kẹt cứng
    if (wHealth[i].isStalled) {
      wpid[i].pwmOutput = max(0, wpid[i].pwmOutput - 50);
      wpid[i].integral = 0;
    }

    // Xác định chiều quay
    int sign = 0;
    if (currentDirection == "ROS")            sign = (rosTargetRpmSigned[i] >= 0.0f) ? 1 : -1;
    else if (currentDirection == "FORWARD")   sign = 1;
    else if (currentDirection == "BACKWARD")  sign = -1;
    else if (currentDirection == "LEFT")      sign = isLeft ? -1 : 1;
    else if (currentDirection == "RIGHT")     sign = isLeft ? 1 : -1;

    slew[i].target = sign * wpid[i].pwmOutput;
  }
}

// ============================================================
//  9. SLEW RATE LIMITER — KHỞI ĐỘNG MỀM (40 Hz)
// ============================================================
void updateSpeedRamp() {
  unsigned long now = millis();
  if (now - lastRampTime < RAMP_INTERVAL_MS) return;
  lastRampTime = now;

  for (int i = 0; i < 4; i++) {
    float &cur  = slew[i].current;
    int    tgt  = slew[i].target;
    float &step = slew[i].step;

    if (tgt == 0) {
      step = constrain(abs(cur) / 8.0f, RAMP_STEP_MIN, RAMP_STEP_STOP_MAX);
    } else {
      if (currentDirection == "ROS") {
        step = 25.0f; // Chế độ ROS: bám sát lệnh điều khiển của Pi, giảm trễ đáp ứng
      } else {
        step = constrain(abs((float)tgt - cur) / 40.0f, RAMP_STEP_MIN, RAMP_STEP_MAX);
      }
    }

    if (abs((float)tgt - cur) <= step) cur = (float)tgt;
    else if (cur < (float)tgt)         cur += step;
    else                               cur -= step;
  }

  writeSingleDrive(DRV1_RPWM, DRV1_LPWM, CH_DRV1_F, CH_DRV1_R, (int)slew[0].current, INV_DRV1);
  writeSingleDrive(DRV2_RPWM, DRV2_LPWM, CH_DRV2_F, CH_DRV2_R, (int)slew[1].current, INV_DRV2);
  writeSingleDrive(DRV3_RPWM, DRV3_LPWM, CH_DRV3_F, CH_DRV3_R, (int)slew[2].current, INV_DRV3);
  writeSingleDrive(DRV4_RPWM, DRV4_LPWM, CH_DRV4_F, CH_DRV4_R, (int)slew[3].current, INV_DRV4);
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
        isMoving = true;
        currentDirection = "ROS";
        for (int i = 0; i < 4; i++) {
          rosTargetRpmSigned[i] = r[i];
          wpid[i].targetRPM = fabsf(r[i]);
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
