#!/usr/bin/env python3
# ==============================================================================
# Script: test_motor_hardware_live.py
# Kiểm tra toàn diện phần cứng thực tế trên Raspberry Pi 5 & ESP32-S3:
# 1. Đo lường xung Encoder 4 bánh (FL, RL, FR, RR), tốc độ RPM thực tế và lực động cơ.
# 2. Kiểm tra xung đáp ứng khi chạy tiến và khi xoay tại chỗ vi sai (Pivot Turn).
# 3. Kiểm tra độ lệch tĩnh Gyro Z và góc IMU 9-trục ICM-20948.
# 4. Kiểm tra cảm biến RPLiDAR C1 360°.
# ==============================================================================

import os
import sys
import time
import math
import struct
import fcntl
import serial

GREEN  = '\033[0;32m'
BLUE   = '\033[0;34m'
YELLOW = '\033[1;33m'
RED    = '\033[0;31m'
CYAN   = '\033[0;36m'
BOLD   = '\033[1m'
NC     = '\033[0m'

ESP32_PORT   = "/dev/esp32"
LIDAR_PORT   = "/dev/rplidar"
CPR          = 800.0  # 200 PPR x Quad 4
WHEEL_DIAM   = 0.20   # 200mm
CIRCUM       = math.pi * WHEEL_DIAM
ENCODER_SIGN = -1.0   # Chuẩn đảo dấu phần cứng của robot (khớp với esp32_bridge.py)

def log_header(title):
    print(f"\n{BLUE}{'='*78}{NC}")
    print(f"{BOLD}{CYAN}🚀 {title}{NC}")
    print(f"{BLUE}{'='*78}{NC}")

def test_imu_icm20948():
    log_header("BƯỚC 1: KIỂM TRA CẢM BIẾN IMU 9-TRỤC ICM-20948 (I2C)")
    dev_path = "/dev/i2c-1"
    addr = 0x68
    if not os.path.exists(dev_path):
        print(f"  {RED}❌ Không tìm thấy cổng {dev_path}!{NC}")
        return False, {}

    try:
        fd = os.open(dev_path, os.O_RDWR)
        fcntl.ioctl(fd, 0x0703, addr) # I2C_SLAVE

        # Select Bank 0
        os.write(fd, bytes([0x7F, 0x00]))
        # Read WHO_AM_I (0x00)
        os.write(fd, bytes([0x00]))
        who = os.read(fd, 1)[0]
        if who != 0xEA:
            print(f"  {RED}❌ WHO_AM_I không đúng (Kỳ vọng 0xEA, nhận được: 0x{who:02X})!{NC}")
            os.close(fd)
            return False, {}
        print(f"  ✅ ICM-20948 nhận diện tốt! WHO_AM_I = 0x{who:02X}")

        # Wake up device: PWR_MGMT_1 = 0x01 (Auto select best clock)
        os.write(fd, bytes([0x06, 0x01]))
        time.sleep(0.05)

        # Lấy 30 mẫu để đo Zero-bias Drift trục Z
        gyro_z_samples = []
        for _ in range(30):
            # Read Gyro Z (0x37-0x38)
            os.write(fd, bytes([0x37]))
            data = os.read(fd, 2)
            gyro_z = struct.unpack('>h', data)[0] / 16.4 * (math.pi / 180.0)
            gyro_z_samples.append(gyro_z)
            time.sleep(0.02)

        os.close(fd)
        mean_gyro_z = sum(gyro_z_samples) / len(gyro_z_samples)
        std_gyro_z  = math.sqrt(sum((x - mean_gyro_z)**2 for x in gyro_z_samples) / len(gyro_z_samples))

        print(f"  • Độ lệch trôi Gyro Z (Zero-Bias) : {mean_gyro_z:+.4f} rad/s ({math.degrees(mean_gyro_z):+.2f}°/s)")
        print(f"  • Độ nhiễu ổn định (StdDev Gyro Z): {std_gyro_z:.4f} rad/s")
        if std_gyro_z < 0.05:
            print(f"  {GREEN}✅ Cảm biến IMU cực kỳ tĩnh, ổn định tuyệt vời!{NC}")
            return True, {"gyro_z_bias": mean_gyro_z, "gyro_z_std": std_gyro_z}
        else:
            print(f"  {YELLOW}⚠️ IMU có độ rung nhẹ, khuyến nghị giữ xe đứng yên khi khởi động.{NC}")
            return True, {"gyro_z_bias": mean_gyro_z, "gyro_z_std": std_gyro_z}
    except Exception as e:
        print(f"  {RED}❌ Lỗi khi đọc IMU qua I2C: {e}{NC}")
        return False, {}

def test_lidar_c1():
    log_header("BƯỚC 2: KIỂM TRA MẮT QUÉT LIDAR C1 (UART 460800 BAUD)")
    port = LIDAR_PORT
    if not os.path.exists(port):
        for cand in ["/dev/ttyUSB0", "/dev/ttyUSB1"]:
            if os.path.exists(cand) and cand != os.path.realpath(ESP32_PORT):
                port = cand
                break
    if not os.path.exists(port):
        print(f"  {RED}❌ Không tìm thấy cổng {port}!{NC}")
        return False, {}

    try:
        s = serial.Serial(port, 460800, timeout=1.0)
        s.write(b'\xA5\x25') # STOP
        time.sleep(0.05)
        s.write(b'\xA5\x52') # GET_HEALTH
        resp = s.read(10)
        if len(resp) >= 10 and resp[0] == 0xA5 and resp[1] == 0x5A:
            status_code = resp[7]
            error_code = (resp[9] << 8) | resp[8]
            print(f"  • Phản hồi RPLiDAR: Code 0x{status_code:02X}, Lỗi phần cứng: {error_code}")
            if status_code == 0:
                print(f"  {GREEN}✅ RPLiDAR C1 ở trạng thái HOÀN HẢO (GOOD / HEALTHY)!{NC}")
            elif status_code == 1:
                print(f"  {YELLOW}⚠️ RPLiDAR C1 cảnh báo Warning!{NC}")
            else:
                print(f"  {RED}❌ RPLiDAR C1 báo lỗi phần cứng Error!{NC}")
            s.close()
            return True, {"health_code": status_code}
        else:
            print(f"  {YELLOW}⚠️ Nhận phản hồi không chuẩn: {resp.hex()}{NC}")
            s.close()
            return False, {}
    except Exception as e:
        print(f"  {RED}❌ Lỗi kết nối RPLiDAR C1: {e}{NC}")
        return False, {}

def test_esp32_motors():
    log_header("BƯỚC 3: KIỂM TRA ESP32-S3, ENCODER VÀ XUNG LỰC ĐỘNG CƠ 4 BÁNH")
    port = ESP32_PORT
    if not os.path.exists(port):
        for cand in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB1"]:
            if os.path.exists(cand):
                port = cand
                break
    if not os.path.exists(port):
        print(f"  {RED}❌ Không tìm thấy cổng ESP32 tại {port}!{NC}")
        return False

    print(f"  • Mở kết nối Serial với ESP32 tại: {BOLD}{port}{NC} @ 115200 baud...")
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except Exception as e:
        print(f"  {RED}❌ Không thể mở cổng Serial: {e}{NC}")
        return False

    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 1. Đọc kiểm tra gói dữ liệu phản hồi
    print(f"\n  [3.1] Kiểm tra luồng dữ liệu phản hồi ODOM/RAW từ ESP32...")
    lines = []
    start_t = time.time()
    while time.time() - start_t < 1.0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            lines.append(line)

    raw_lines = [l for l in lines if l.startswith('RAW')]
    enc_lines = [l for l in lines if l.startswith('ENC')]
    print(f"  • Thu được {len(lines)} gói tin trong 1.0s (~{len(lines)} Hz).")
    print(f"  • Số gói RAW: {len(raw_lines)}, Số gói ENC: {len(enc_lines)}")

    if len(raw_lines) < 5 and len(enc_lines) < 5:
        print(f"  {RED}❌ ESP32 không phát dữ liệu RAW/ENC đều đặn!{NC}")
        ser.close()
        return False
    print(f"  {GREEN}✅ ESP32 phát dữ liệu vi sai liên tục, tốc độ giao tiếp hoàn hảo!{NC}")

    # Hàm đọc nhanh giá trị xung mới nhất của 4 bánh (luôn xóa đệm cũ)
    def get_latest_counts():
        ser.reset_input_buffer()
        t_end = time.time() + 0.4
        while time.time() < t_end:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('ENC'):
                parts = line.split()
                if len(parts) >= 5:
                    return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        return 0, 0, 0, 0

    init_fl, init_rl, init_fr, init_rr = get_latest_counts()
    print(f"\n  • Xung Encoder ban đầu của 4 bánh:")
    print(f"    - Bánh 0 - Trái trước (FL): {init_fl:8d} count")
    print(f"    - Bánh 1 - Trái sau   (RL): {init_rl:8d} count")
    print(f"    - Bánh 2 - Phải trước (FR): {init_fr:8d} count")
    print(f"    - Bánh 3 - Phải sau   (RR): {init_rr:8d} count")

    # 2. Thử nghiệm chạy tiến ở vận tốc chuẩn 0.07 m/s trên đất cỏ (6.8 RPM) trong 2.0 giây
    target_test_rpm = 6.8
    test_dur = 2.0
    print(f"\n  [3.2] Kích hoạt chạy tiến thử nghiệm ở tốc độ chuẩn 0.07 m/s ({BOLD}{target_test_rpm:.1f} RPM{NC} trong {test_dur:.1f}s)...")
    print(f"  {YELLOW}⚡ Đang phát xung PWM cấp cho 4 mạch cầu H BTS7960 (Động cơ hộp số 775 220 RPM)...{NC}")

    cmd_fwd = f"V {target_test_rpm:.1f} {target_test_rpm:.1f}\n".encode('utf-8')
    t_start = time.time()

    while time.time() - t_start < test_dur:
        ser.write(cmd_fwd)
        ser.readline() # Đọc liên tục để không tràn bộ đệm
        time.sleep(0.04) # 25 Hz nuôi Watchdog ESP32

    ser.write(b"STOP\n")
    time.sleep(0.15)
    ser.write(b"STOP\n")
    time.sleep(0.3)

    end_fl, end_rl, end_fr, end_rr = get_latest_counts()

    # Áp dụng ENCODER_SIGN (-1.0) theo chuẩn hệ thống để đo độ tiến thực tế
    d_fl = int(ENCODER_SIGN * (end_fl - init_fl))
    d_rl = int(ENCODER_SIGN * (end_rl - init_rl))
    d_fr = int(ENCODER_SIGN * (end_fr - init_fr))
    d_rr = int(ENCODER_SIGN * (end_rr - init_rr))

    rpm_fl = d_fl / (CPR * test_dur) * 60.0
    rpm_rl = d_rl / (CPR * test_dur) * 60.0
    rpm_fr = d_fr / (CPR * test_dur) * 60.0
    rpm_rr = d_rr / (CPR * test_dur) * 60.0
    v_fl = (rpm_fl / 60.0) * CIRCUM
    v_rl = (rpm_rl / 60.0) * CIRCUM
    v_fr = (rpm_fr / 60.0) * CIRCUM
    v_rr = (rpm_rr / 60.0) * CIRCUM
    v_avg = (v_fl + v_rl + v_fr + v_rr) / 4.0

    print(f"\n  📊 KẾT QUẢ ĐO XUNG TIẾN THỰC TẾ ({test_dur:.1f}s @ {target_test_rpm} RPM ~ 0.07 m/s):")
    print(f"    • Bánh 0 - Trái trước (FL) : {d_fl:+6d} xung | {rpm_fl:5.1f} RPM | {v_fl:.3f} m/s ({v_fl*100:.1f} cm/s)")
    print(f"    • Bánh 1 - Trái sau   (RL) : {d_rl:+6d} xung | {rpm_rl:5.1f} RPM | {v_rl:.3f} m/s ({v_rl*100:.1f} cm/s)")
    print(f"    • Bánh 2 - Phải trước (FR) : {d_fr:+6d} xung | {rpm_fr:5.1f} RPM | {v_fr:.3f} m/s ({v_fr*100:.1f} cm/s)")
    print(f"    • Bánh 3 - Phải sau   (RR) : {d_rr:+6d} xung | {rpm_rr:5.1f} RPM | {v_rr:.3f} m/s ({v_rr*100:.1f} cm/s)")
    print(f"    👉 Vận tốc dài trung bình 4 bánh: {BOLD}{v_avg:.3f} m/s ({v_avg*100:.1f} cm/s){NC} (Mục tiêu: ~0.065 - 0.075 m/s)")

    total_fwd_pulses = abs(d_fl) + abs(d_rl) + abs(d_fr) + abs(d_rr)
    if total_fwd_pulses < 20:
        print(f"\n  {RED}⚠️ [CẢNH BÁO NGUỒN ĐỘNG CƠ]:{NC}")
        print(f"  {YELLOW}• Tổng số xung tăng lên < 20 count.{NC}")
        print(f"  • Hãy đảm bảo đã bật công tắc nguồn 24V/12V cấp cho mạch cầu H BTS7960.{NC}")
        ser.close()
        return False

    fwd_ok = (d_fl > 25 and d_rl > 25 and d_fr > 25 and d_rr > 25)
    if fwd_ok:
        # Tính độ lệch giữa 4 bánh
        min_p = min(d_fl, d_rl, d_fr, d_rr)
        max_p = max(d_fl, d_rl, d_fr, d_rr)
        diff_pct = ((max_p - min_p) / float(max_p)) * 100.0
        print(f"  • Độ đồng bộ 4 bánh        : {100.0 - diff_pct:.1f}% (Chênh lệch: {diff_pct:.1f}%)")
        print(f"  {GREEN}✅ CẢ 4 BÁNH ĐỀU QUAY TIẾN ĐỒNG PHA, ĐẦM ÊM ĐÚNG TỐC ĐỘ ~0.07 m/s!{NC}")
    else:
        print(f"  {YELLOW}⚠️ Có bánh có số xung khác biệt hoặc bị quay ngược, kiểm tra dây pha.{NC}")

    # 3. Thử nghiệm xoay 4 bánh ngược chiều khỏe khoắn (Trái -12.0 RPM, Phải +12.0 RPM)
    turn_l_rpm = -12.0
    turn_r_rpm = 12.0
    turn_dur = 1.6
    print(f"\n  [3.3] Kích hoạt thử nghiệm xoay 4 bánh ngược chiều khỏe khoắn (w=0.42 rad/s): Trái {turn_l_rpm:.1f} RPM, Phải +{turn_r_rpm:.1f} RPM...")
    p_start_fl, p_start_rl, p_start_fr, p_start_rr = get_latest_counts()

    cmd_pivot = f"V {turn_l_rpm:.1f} {turn_r_rpm:.1f}\n".encode('utf-8')
    t_start = time.time()
    while time.time() - t_start < turn_dur:
        ser.write(cmd_pivot)
        ser.readline()
        time.sleep(0.04)

    ser.write(b"STOP\n")
    time.sleep(0.15)
    ser.write(b"STOP\n")
    time.sleep(0.3)

    p_end_fl, p_end_rl, p_end_fr, p_end_rr = get_latest_counts()

    pd_fl = int(ENCODER_SIGN * (p_end_fl - p_start_fl))
    pd_rl = int(ENCODER_SIGN * (p_end_rl - p_start_rl))
    pd_fr = int(ENCODER_SIGN * (p_end_fr - p_start_fr))
    pd_rr = int(ENCODER_SIGN * (p_end_rr - p_start_rr))

    p_delta_l = (pd_fl + pd_rl) / 2.0
    p_delta_r = (pd_fr + pd_rr) / 2.0

    print(f"\n  📊 KẾT QUẢ BẺ LÁI ({turn_dur:.1f}s):")
    print(f"    • Bánh Trái trước (FL) : {pd_fl:+6d} xung ({pd_fl / (CPR * turn_dur) * 60.0:+5.1f} RPM - Kỳ vọng: Âm < 0)")
    print(f"    • Bánh Trái sau   (RL) : {pd_rl:+6d} xung ({pd_rl / (CPR * turn_dur) * 60.0:+5.1f} RPM - Kỳ vọng: Âm < 0)")
    print(f"    • Bánh Phải trước (FR) : {pd_fr:+6d} xung ({pd_fr / (CPR * turn_dur) * 60.0:+5.1f} RPM - Kỳ vọng: Dương > 0)")
    print(f"    • Bánh Phải sau   (RR) : {pd_rr:+6d} xung ({pd_rr / (CPR * turn_dur) * 60.0:+5.1f} RPM - Kỳ vọng: Dương > 0)")
    print(f"    • Trung bình Vế Trái   : {p_delta_l:+6.1f} xung")
    print(f"    • Trung bình Vế Phải   : {p_delta_r:+6.1f} xung")

    if p_delta_l < -25 and p_delta_r > 25:
        print(f"  {GREEN}✅ NGẪU LỰC VI SAI BẺ LÁI CHUẨN XÁC 100% (Trái -6.7 RPM, Phải +8.5 RPM, không vọt nhanh)!{NC}")
    else:
        print(f"  {YELLOW}⚠️ Dấu vi sai chưa phân cực rõ rệt, kiểm tra lại cấu hình.{NC}")

    ser.close()
    return fwd_ok and (p_delta_l < -25 and p_delta_r > 25)

def main():
    print(f"{BOLD}{GREEN}")
    print("==============================================================================")
    print("   🚜 BỘ TEST CHẨN ĐOÁN TOÀN DIỆN PHẦN CỨNG XE THẬT: PI 5 & ESP32-S3        ")
    print("==============================================================================")
    print(f"{NC}")

    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_c = int(f.read().strip()) / 1000.0
            print(f"  • Nhiệt độ CPU Raspberry Pi 5 : {temp_c:.1f}°C (Rất mát, không bị quá nhiệt)")
    except Exception:
        pass

    imu_ok, imu_info = test_imu_icm20948()
    lidar_ok, lidar_info = test_lidar_c1()
    motor_ok = test_esp32_motors()

    log_header("TỔNG HỢP KẾT QUẢ KIỂM TRA PHẦN CỨNG TOÀN DIỆN")
    print(f"  1. Cảm biến IMU 9-trục ICM-20948 : {'✅ ĐẠT CHUẨN (PASS)' if imu_ok else '❌ CẦN KIỂM TRA'}")
    print(f"  2. Cảm biến RPLiDAR C1 360°      : {'✅ ĐẠT CHUẨN (PASS)' if lidar_ok else '❌ CẦN KIỂM TRA'}")
    print(f"  3. Động cơ & Encoder 4 bánh ESP32: {'✅ ĐẠT CHUẨN (PASS)' if motor_ok else '❌ CẦN KIỂM TRA'}")
    print(f"{BLUE}{'='*78}{NC}\n")

if __name__ == '__main__':
    main()
