#!/usr/bin/env python3
"""
==============================================================================
  ROS 2 Node: encoder_node (Package: encoder_odom)
  
  Mục đích:
  - Giao tiếp USB Serial với ESP32 nhận luồng dữ liệu 4 encoder Quadrature X4:
      Format: RAW <sequence> <timestamp_us> <FL> <FR> <RL> <RR>
      (Hoặc CSV: <sequence>,<timestamp_us>,<FL>,<FR>,<RL>,<RR>)
  - Kiểm tra rớt gói (Packet Loss Check) qua sequence và timestamp phần cứng.
  - Tính toán quãng đường, vận tốc độc lập cho từng bánh: FL, FR, RL, RR.
  - Lớp kiểm tra chéo & cách ly lỗi cảm biến bất thường:
      + Kiểm tra vế Trái: FL <-> RL
      + Kiểm tra vế Phải: FR <-> RR
      + Tự động loại bỏ bánh bị nhiễu/nhảy số đột biến, tránh làm sai lệch Odometry.
  - Động học 4 bánh Skid-Steer vi sai:
      + V_left  = (V_FL + V_RL) / 2.0
      + V_right = (V_FR + V_RR) / 2.0
      + Vx = (V_left + V_right) / 2.0
      + Wz = (V_right - V_left) / track_width
  - Tích phân Dead Reckoning và xuất chuẩn nav_msgs/Odometry lên topic /wheel/odom.
  - Nhận lệnh điều khiển /cmd_vel chuyển đổi RPM gửi xuống ESP32 nuôi Watchdog.
==============================================================================
"""

import os
import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, TransformStamped
from std_msgs.msg import String as StringMsg
import tf2_ros

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class EncoderNode(Node):
    def __init__(self):
        super().__init__('encoder_node')

        # ── 1. Khai báo tham số ROS 2 ─────────────────────────────────────
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_diameter', 0.20)      # Đường kính bánh (m): D = 0.2m -> Chu vi ~ 0.6283m
        self.declare_parameter('track_width', 0.58)         # Khoảng cách tâm 2 vế bánh trái-phải (m)
        self.declare_parameter('encoder_ppr', 200)          # Xung/vòng đơn kênh của encoder (PPR)
        self.declare_parameter('quadrature', 4)             # Nhân 4 ngắt Quadrature X4 -> 800 CPR
        self.declare_parameter('gear_ratio', 1.0)           # Tỷ số truyền sau hộp số (1.0 nếu đo trục bánh)
        self.declare_parameter('odom_topic', '/wheel/odom') # Topic Odometry đầu ra
        self.declare_parameter('publish_tf', False)         # Tự phát TF odom -> base_footprint (thường False khi dùng EKF)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('abnormal_diff_threshold', 0.6)  # Ngưỡng lệch tốc độ tối đa giữa 2 bánh cùng bên (m/s)
        self.declare_parameter('max_accel_threshold', 8.0)       # Ngưỡng gia tốc vật lý tối đa cho phép (m/s^2)
        self.declare_parameter('cmd_vel_timeout', 0.60)         # Timeout lệnh cmd_vel (0.60s giúp chống rớt gói/lag mạng Wi-Fi)
        self.declare_parameter('min_moving_rpm', 6.7)           # Sàn RPM chuẩn 0.07 m/s (~6.68 RPM bánh D=0.20m) cho động cơ 775 220RPM
        self.declare_parameter('rpm_scale', 1.05)               # Hệ số bù trượt đất mềm (+5% -> ~7.0 RPM để đạt chuẩn 0.07 m/s)

        # ── 2. Lấy giá trị tham số ────────────────────────────────────────
        self.port = self.get_parameter('serial_port').value
        self.baud = int(self.get_parameter('baudrate').value)
        self.wheel_d = float(self.get_parameter('wheel_diameter').value)
        self.track_width = float(self.get_parameter('track_width').value)
        self.encoder_ppr = int(self.get_parameter('encoder_ppr').value)
        self.quadrature = int(self.get_parameter('quadrature').value)
        self.gear_ratio = float(self.get_parameter('gear_ratio').value)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.abnormal_diff_threshold = float(self.get_parameter('abnormal_diff_threshold').value)
        self.max_accel_threshold = float(self.get_parameter('max_accel_threshold').value)
        self.cmd_vel_timeout = float(self.get_parameter('cmd_vel_timeout').value)
        self.min_moving_rpm = float(self.get_parameter('min_moving_rpm').value)
        self.rpm_scale = float(self.get_parameter('rpm_scale').value)

        # ── 3. Hằng số động học bánh xe ────────────────────────────────────
        self.wheel_circ = math.pi * self.wheel_d
        self.cpr = float(self.encoder_ppr * self.quadrature * self.gear_ratio)
        self.distance_per_count = self.wheel_circ / self.cpr

        # ── 4. Trạng thái theo dõi xung & Phần cứng ──────────────────────
        self.ser = None
        self._rx_buf = ''
        self._last_seq = None
        self._last_timestamp_us = None
        self._last_fl = None
        self._last_fr = None
        self._last_rl = None
        self._last_rr = None

        self._last_v_fl = 0.0
        self._last_v_fr = 0.0
        self._last_v_rl = 0.0
        self._last_v_rr = 0.0
        self._last_v_left = 0.0
        self._last_v_right = 0.0

        self.dropped_packets_total = 0

        # ── 5. Trạng thái Odometry Dead Reckoning ───────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.last_ros_time = self.get_clock().now()

        # ── 6. Trạng thái điều khiển Motor (cmd_vel) ────────────────────────
        self.target_rpm_left = 0.0
        self.target_rpm_right = 0.0
        self.last_cmd_vel_time = 0.0

        # ── 7. Publishers, Subscribers & TF Broadcaster ────────────────────
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.wheel_status_pub = self.create_publisher(StringMsg, '/wheel/status', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_tf else None
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Timer chu kỳ 20 Hz đọc Serial, tích phân Odometry và duy trì nhịp lệnh
        self.timer = self.create_timer(0.05, self.update_loop)

        # Khởi tạo kết nối Serial
        self.init_serial()

        self.get_logger().info(
            f'🚀 [encoder_node] Khởi động thành công! [Port: {self.port} @ {self.baud}] '
            f'[Wheel D: {self.wheel_d}m, Track: {self.track_width}m] '
            f'[CPR: {self.cpr:.0f} (PPR:{self.encoder_ppr} x Quad:{self.quadrature} x Gear:{self.gear_ratio})] '
            f'-> Output topic: {self.odom_topic}'
        )

    def init_serial(self):
        """Khởi tạo hoặc tự động tìm cổng Serial kết nối tới ESP32."""
        if not SERIAL_AVAILABLE:
            self.get_logger().error('Thư viện pyserial chưa được cài đặt! Hãy chạy: pip install pyserial')
            return

        candidate_ports = [self.port, '/dev/ttyACM0', '/dev/esp32', '/dev/ttyACM1', '/dev/ttyUSB1', '/dev/ttyUSB0']
        seen = set()
        ports_to_try = [p for p in candidate_ports if p and (p not in seen and not seen.add(p))]

        for p in ports_to_try:
            if not os.path.exists(p):
                continue
            try:
                self.ser = serial.Serial(p, self.baud, timeout=0.02)
                self.port = p
                self.get_logger().info(f'✅ Đã mở thành công cổng Serial ESP32: {self.port} ở {self.baud} baud')
                return
            except Exception as e:
                self.get_logger().warn(f'Không thể mở cổng {p}: {e}')

        self.get_logger().warn('⚠️ Chưa tìm thấy cổng Serial ESP32. Sẽ tự động kết nối lại trong vòng lặp...')

    def cmd_vel_callback(self, msg: Twist):
        """
        Nhận /cmd_vel (linear.x, angular.z) từ Navigation2 / Teleop.
        Quy đổi ra RPM 2 vế và cập nhật mục tiêu cho vòng lặp điều khiển 20 Hz.
        Không gửi trực tiếp ở đây để tránh xung đột bus Serial (Double-Send).
        """
        v = msg.linear.x
        w = msg.angular.z

        if abs(v) < 0.005 and abs(w) < 0.005:
            self.target_rpm_left = 0.0
            self.target_rpm_right = 0.0
        elif abs(v) < 0.02:
            pivot_rpm = max(11.5, abs(w * self.track_width / 2.0 * 60.0 / self.wheel_circ) * self.rpm_scale)
            self.target_rpm_left = -pivot_rpm if w > 0 else pivot_rpm
            self.target_rpm_right = pivot_rpm if w > 0 else -pivot_rpm
        else:
            v_left = v - (w * self.track_width / 2.0)
            v_right = v + (w * self.track_width / 2.0)
            rpm_l = ((v_left * 60.0) / self.wheel_circ) * self.rpm_scale
            rpm_r = ((v_right * 60.0) / self.wheel_circ) * self.rpm_scale
            avg_rpm = (rpm_l + rpm_r) / 2.0
            diff_rpm = (rpm_r - rpm_l) / 2.0
            sign_fwd = 1.0 if avg_rpm >= 0 else -1.0
            eff_fwd = sign_fwd * max(self.min_moving_rpm, abs(avg_rpm))
            rpm_l = eff_fwd - diff_rpm
            rpm_r = eff_fwd + diff_rpm
            if v > 0.02 and abs(w) < 0.40:
                min_inner_rpm = min(4.8, self.min_moving_rpm * 0.72)
                min_side_rpm = min(rpm_l, rpm_r)
                if min_side_rpm < min_inner_rpm:
                    shift_rpm = min_inner_rpm - min_side_rpm
                    rpm_l += shift_rpm
                    rpm_r += shift_rpm
            self.target_rpm_left = rpm_l
            self.target_rpm_right = rpm_r
        self.last_cmd_vel_time = time.time()

    def _send_motor_cmd(self, rpm_l: float, rpm_r: float):
        """Gửi lệnh vận tốc tường minh 4 bánh 'V <FL> <RL> <FR> <RR>' xuống ESP32."""
        if self.ser and self.ser.is_open:
            # Gửi tường minh cả 4 bánh để ESP32 phân giải độc lập, tránh nhầm vế
            cmd = f'V {rpm_l:.1f} {rpm_l:.1f} {rpm_r:.1f} {rpm_r:.1f}\n'
            try:
                self.ser.write(cmd.encode('utf-8'))
            except Exception as e:
                self.get_logger().warn(f'Lỗi gửi Serial xuống ESP32: {e}')

    def update_loop(self):
        """Vòng lặp chính 20 Hz: Đọc Serial non-blocking, xử lý Odometry và gửi heartbeat điều khiển."""
        now = self.get_clock().now()

        # Kiểm tra kết nối lại nếu mất Serial
        if not self.ser or not self.ser.is_open:
            self.init_serial()
            return

        # Duy trì lệnh gửi định kỳ nuôi Watchdog an toàn của ESP32 (Chu kỳ chuẩn 20 Hz / 50ms)
        # Nếu quá cmd_vel_timeout (0.60s) không có lệnh /cmd_vel mới thì tự động về 0 RPM để dừng xe an toàn
        if time.time() - self.last_cmd_vel_time > self.cmd_vel_timeout:
            self.target_rpm_left = 0.0
            self.target_rpm_right = 0.0
        self._send_motor_cmd(self.target_rpm_left, self.target_rpm_right)

        # Đọc dữ liệu từ Serial buffer
        try:
            if self.ser.in_waiting > 0:
                raw_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                self._rx_buf += raw_data

                while '\n' in self._rx_buf:
                    line, self._rx_buf = self._rx_buf.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._parse_line(line)
        except Exception as e:
            self.get_logger().warn(f'Lỗi đọc Serial: {e}')
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _parse_line(self, line: str):
        """
        Phân tích cú pháp gói tin encoder từ ESP32:
        Hỗ trợ các định dạng:
        1. RAW <sequence> <timestamp_us> <FL> <FR> <RL> <RR>
        2. <sequence>,<timestamp_us>,<FL>,<FR>,<RL>,<RR> (CSV)
        3. RAW <timestamp_us> <FL> <FR> <RL> <RR> (Tương thích ngược)
        """
        seq = None
        timestamp_us = None
        fl = fr = rl = rr = None

        if line.startswith('RAW'):
            # Chuẩn Space hoặc Comma
            cleaned = line.replace(',', ' ')
            parts = cleaned.split()
            # Kiểm tra định dạng có Sequence: "RAW seq t_us fl fr rl rr" (7 phần tử)
            if len(parts) >= 7:
                try:
                    seq = int(parts[1])
                    timestamp_us = int(parts[2])
                    fl = int(parts[3])
                    fr = int(parts[4])
                    rl = int(parts[5])
                    rr = int(parts[6])
                except ValueError:
                    return
            # Tương thích ngược: "RAW t_us fl fr rl rr" (6 phần tử)
            elif len(parts) >= 6:
                try:
                    seq = (self._last_seq + 1) if self._last_seq is not None else 0
                    timestamp_us = int(parts[1])
                    fl = int(parts[2])
                    fr = int(parts[3])
                    rl = int(parts[4])
                    rr = int(parts[5])
                except ValueError:
                    return
        elif ',' in line:
            # Định dạng thuần CSV: seq,timestamp,fl,fr,rl,rr
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    seq = int(parts[0].strip())
                    timestamp_us = int(parts[1].strip())
                    fl = int(parts[2].strip())
                    fr = int(parts[3].strip())
                    rl = int(parts[4].strip())
                    rr = int(parts[5].strip())
                except ValueError:
                    return
        else:
            # Dòng log / debug của ESP32 (ví dụ # [WATCHDOG], IP, etc.) -> Bỏ qua
            return

        if timestamp_us is not None and fl is not None:
            self._process_encoder_data(seq, timestamp_us, fl, fr, rl, rr)

    def _process_encoder_data(self, seq: int, t_us: int, fl: int, fr: int, rl: int, rr: int):
        """
        Xử lý tính toán vi sai, kiểm tra rớt gói, kiểm tra chéo và tính Odometry.
        """
        if self._last_timestamp_us is None:
            # Gói đầu tiên: Lưu lại mốc trạng thái ban đầu
            self._last_seq = seq
            self._last_timestamp_us = t_us
            self._last_fl = fl
            self._last_fr = fr
            self._last_rl = rl
            self._last_rr = rr
            return

        # ── 1. Kiểm tra rớt gói (Packet Loss Check) qua Sequence ────────
        if self._last_seq is not None and seq is not None:
            delta_seq = seq - self._last_seq
            if delta_seq > 1:
                dropped = delta_seq - 1
                self.dropped_packets_total += dropped
                self.get_logger().warn(
                    f'⚠️ [PACKET DROP] Phát hiện mất {dropped} gói tin giữa Seq {self._last_seq} -> {seq} '
                    f'(Tổng mất: {self.dropped_packets_total})', throttle_duration_sec=2.0
                )

        # ── 2. Tính khoảng thời gian dt từ Timestamp phần cứng của ESP32 ──
        # Xử lý an toàn hiện tượng tràn số nguyên không dấu 32-bit của hàm micros() (sau ~71 phút)
        dt_us = t_us - self._last_timestamp_us
        if dt_us < 0:
            dt_us += (1 << 32)
        dt = dt_us / 1e6  # Chuyển đổi ra giây

        # Kiểm tra chu kỳ lấy mẫu có hợp lý không (từ 5ms đến 400ms)
        if dt < 0.005 or dt > 0.400:
            self.get_logger().warn(f'Bỏ qua gói tin do chu kỳ dt bất thường: {dt:.4f}s', throttle_duration_sec=2.0)
            self._last_seq = seq
            self._last_timestamp_us = t_us
            self._last_fl = fl
            self._last_fr = fr
            self._last_rl = rl
            self._last_rr = rr
            return

        # ── 3. Tính Δcount từng bánh độc lập ──────────────────────────────
        delta_fl = fl - self._last_fl
        delta_fr = fr - self._last_fr
        delta_rl = rl - self._last_rl
        delta_rr = rr - self._last_rr

        # Cập nhật ngay mốc trước cho lần sau
        self._last_seq = seq
        self._last_timestamp_us = t_us
        self._last_fl = fl
        self._last_fr = fr
        self._last_rl = rl
        self._last_rr = rr

        # ── 4. Từ Δcount -> Quãng đường từng bánh (m) ──────────────────────
        dist_fl = delta_fl * self.distance_per_count
        dist_fr = delta_fr * self.distance_per_count
        dist_rl = delta_rl * self.distance_per_count
        dist_rr = delta_rr * self.distance_per_count

        # ── 5. Từ Quãng đường -> Vận tốc từng bánh (m/s) ───────────────────
        v_fl = dist_fl / dt
        v_fr = dist_fr / dt
        v_rl = dist_rl / dt
        v_rr = dist_rr / dt

        # ── 6. Lớp kiểm tra chéo & cách ly lỗi cảm biến bất thường ────────
        # Kiểm tra vế Trái (FL <-> RL)
        diff_left = abs(v_fl - v_rl)
        fl_anomalous = False
        rl_anomalous = False

        if diff_left > self.abnormal_diff_threshold:
            # Hai bánh vế trái lệch nhau quá lớn: Kiểm tra xem bánh nào có gia tốc bất thường
            a_fl = abs(v_fl - self._last_v_fl) / dt
            a_rl = abs(v_rl - self._last_v_rl) / dt

            if a_fl > self.max_accel_threshold and a_rl <= self.max_accel_threshold:
                fl_anomalous = True
                self.get_logger().warn(
                    f'⚠️ [ANOMALY ISOLATED] Bánh FL nhảy số bất thường ({v_fl:.2f} m/s)! Sử dụng RL ({v_rl:.2f} m/s) làm đại diện vế Trái.',
                    throttle_duration_sec=2.0
                )
            elif a_rl > self.max_accel_threshold and a_fl <= self.max_accel_threshold:
                rl_anomalous = True
                self.get_logger().warn(
                    f'⚠️ [ANOMALY ISOLATED] Bánh RL nhảy số bất thường ({v_rl:.2f} m/s)! Sử dụng FL ({v_fl:.2f} m/s) làm đại diện vế Trái.',
                    throttle_duration_sec=2.0
                )
            else:
                self.get_logger().warn(
                    f'⚠️ [SLIP DETECTED] Vế Trái có chênh lệch tốc độ: FL={v_fl:.2f}, RL={v_rl:.2f} (Δ={diff_left:.2f} m/s)',
                    throttle_duration_sec=2.0
                )

        if fl_anomalous:
            v_left = v_rl
        elif rl_anomalous:
            v_left = v_fl
        else:
            v_left = (v_fl + v_rl) / 2.0

        # Kiểm tra vế Phải (FR <-> RR)
        diff_right = abs(v_fr - v_rr)
        fr_anomalous = False
        rr_anomalous = False

        if diff_right > self.abnormal_diff_threshold:
            a_fr = abs(v_fr - self._last_v_fr) / dt
            a_rr = abs(v_rr - self._last_v_rr) / dt

            if a_fr > self.max_accel_threshold and a_rr <= self.max_accel_threshold:
                fr_anomalous = True
                self.get_logger().warn(
                    f'⚠️ [ANOMALY ISOLATED] Bánh FR nhảy số bất thường ({v_fr:.2f} m/s)! Sử dụng RR ({v_rr:.2f} m/s) làm đại diện vế Phải.',
                    throttle_duration_sec=2.0
                )
            elif a_rr > self.max_accel_threshold and a_fr <= self.max_accel_threshold:
                rr_anomalous = True
                self.get_logger().warn(
                    f'⚠️ [ANOMALY ISOLATED] Bánh RR nhảy số bất thường ({v_rr:.2f} m/s)! Sử dụng FR ({v_fr:.2f} m/s) làm đại diện vế Phải.',
                    throttle_duration_sec=2.0
                )
            else:
                self.get_logger().warn(
                    f'⚠️ [SLIP DETECTED] Vế Phải có chênh lệch tốc độ: FR={v_fr:.2f}, RR={v_rr:.2f} (Δ={diff_right:.2f} m/s)',
                    throttle_duration_sec=2.0
                )

        if fr_anomalous:
            v_right = v_rr
        elif rr_anomalous:
            v_right = v_fr
        else:
            v_right = (v_fr + v_rr) / 2.0

        # Cập nhật lịch sử vận tốc từng bánh
        self._last_v_fl = v_fl
        self._last_v_fr = v_fr
        self._last_v_rl = v_rl
        self._last_v_rr = v_rr
        self._last_v_left = v_left
        self._last_v_right = v_right

        # ── 7. Động học 4 bánh Skid-Steer vi sai (Giữ độc lập Trái / Phải) ──
        # Vận tốc tiến tâm xe:
        vx = (v_left + v_right) / 2.0
        # Vận tốc góc quay quanh trục Z:
        wz = (v_right - v_left) / self.track_width

        # Bộ lọc an toàn: Giới hạn vật lý tối đa của robot
        if abs(vx) > 3.0:
            vx = math.copysign(3.0, vx)
        if abs(wz) > 5.0:
            wz = math.copysign(5.0, wz)

        self.vx = vx
        self.wz = wz

        # ── 8. Tích phân Dead Reckoning (X, Y, Yaw) ────────────────────────
        delta_x = (vx * math.cos(self.yaw)) * dt
        delta_y = (vx * math.sin(self.yaw)) * dt
        delta_yaw = wz * dt

        self.x += delta_x
        self.y += delta_y
        self.yaw += delta_yaw

        # Chuẩn hóa góc quay Yaw trong khoảng [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        # ── 9. Xuất bản Odometry (/wheel/odom) & Trạng thái 4 bánh (/wheel/status) ──
        self._publish_odometry()

        # Xuất bản chuỗi trạng thái chi tiết 4 bánh để người dùng theo dõi trực tiếp
        status_msg = StringMsg()
        status_msg.data = (
            f"Seq:{seq} | FL:{fl} ({v_fl:+.2f}m/s) | FR:{fr} ({v_fr:+.2f}m/s) | "
            f"RL:{rl} ({v_rl:+.2f}m/s) | RR:{rr} ({v_rr:+.2f}m/s) | "
            f"Vx:{vx:+.2f}m/s | Wz:{wz:+.2f}rad/s"
        )
        self.wheel_status_pub.publish(status_msg)

        # In log định kỳ 1 giây/lần lên màn hình Terminal
        self.get_logger().info(
            f"📊 [4 ENCODERS] FL={fl} FR={fr} RL={rl} RR={rr} | Vx={vx:+.2f}m/s, Wz={wz:+.2f}rad/s",
            throttle_duration_sec=1.0
        )

    def _publish_odometry(self):
        """Tạo và xuất bản thông điệp nav_msgs/Odometry và TF (nếu bật)."""
        now = self.get_clock().now()

        # Chuyển đổi góc Yaw sang Quaternion (quaternion_from_euler)
        half_yaw = self.yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        # Vị trí (Pose)
        odom_msg.pose.pose.position.x = float(self.x)
        odom_msg.pose.pose.position.y = float(self.y)
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = float(qz)
        odom_msg.pose.pose.orientation.w = float(qw)

        # Ma trận hiệp phương sai sai số Vị trí (Pose Covariance)
        # [x, y, z, roll, pitch, yaw]
        odom_msg.pose.covariance = [
            0.01, 0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.01, 0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  999.0,0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 999.0,0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 999.0,0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.03
        ]

        # Vận tốc (Twist)
        odom_msg.twist.twist.linear.x = float(self.vx)
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = float(self.wz)

        # Ma trận hiệp phương sai Vận tốc (Twist Covariance)
        odom_msg.twist.covariance = [
            0.005, 0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.005, 0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  999.0,0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 999.0,0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 999.0,0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.01
        ]

        self.odom_pub.publish(odom_msg)

        # Xuất bản Transform TF odom -> base_footprint (nếu bật publish_tf)
        if self.publish_tf and self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = float(self.x)
            t.transform.translation.y = float(self.y)
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.ser and node.ser.is_open:
            try:
                node.ser.write(b'V 0.0 0.0\n')
                node.ser.close()
            except Exception:
                pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
