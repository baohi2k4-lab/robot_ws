#!/usr/bin/env python3
"""
ESP32 Hardware Bridge Node for ROS 2.
Connects ROS 2 high-level navigation/AI to ESP32 motor PID & Encoder controller.

Roles:
1. Downlink: Converts /cmd_vel (v, w) -> wheel Target RPMs -> Sends to ESP32 via Serial/WiFi.
2. Uplink: Receives wheel encoder speeds/distances from ESP32 -> Computes Odometry (x, y, yaw) -> Publishes /odom & TF.
3. Fallback: If ESP32 is not connected, operates in graceful Mock/Open-Loop mode.
"""

import os
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String as StringMsg
import tf2_ros

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ESP32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('connection_mode', 'serial')  # 'serial', 'wifi', 'mock'
        self.declare_parameter('serial_port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('esp32_ip', '192.168.1.100')
        self.declare_parameter('wheel_diameter', 0.20)  # meters (from code0409.ino)
        self.declare_parameter('wheel_base', 0.58)      # meters (track width between left & right wheels)
        self.declare_parameter('encoder_ppr', 200)      # Pulses per revolution per channel
        self.declare_parameter('quadrature', 4)         # Quadrature factor (1, 2, or 4 for X4)
        self.declare_parameter('gear_ratio', 1.0)       # Gearbox ratio (1.0 if encoder is after gearbox)
        self.declare_parameter('odom_topic', '/wheel/odom')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('min_moving_rpm', 6.8)   # Sàn RPM chuẩn 0.07 m/s (~6.8 RPM bánh D=0.20m) khỏe dư lực vượt mô đất cỏ gồ ghề
        self.declare_parameter('rpm_scale', 1.00)       # Hệ số tỉ lệ 1.00 (kết hợp PID_KI=11.0 trên ESP32 giữ chuẩn 0.07 m/s)
        self.declare_parameter('max_wheel_rpm', 20.0)   # Trần RPM mỗi bánh (~0.21 m/s) đủ lực mạnh cho pivot turn 4 bánh ngược chiều trên đất cỏ gồ ghề
        self.declare_parameter('encoder_sign', -1.0)    # -1.0: Đảo dấu xung encoder chuẩn xác với chiều tiến thực tế
        self.declare_parameter('cmd_vel_timeout', 1.0)  # s — Mất /cmd_vel quá thời gian này (rớt WiFi, laptop treo) -> tự dừng xe

        self.mode = self.get_parameter('connection_mode').value
        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baudrate').value
        self.wheel_d = float(self.get_parameter('wheel_diameter').value)
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.encoder_ppr = int(self.get_parameter('encoder_ppr').value)
        self.quadrature = int(self.get_parameter('quadrature').value)
        self.gear_ratio = float(self.get_parameter('gear_ratio').value)
        self.encoder_cpr = float(self.encoder_ppr * self.quadrature * self.gear_ratio)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.min_moving_rpm = float(self.get_parameter('min_moving_rpm').value)
        self.rpm_scale = float(self.get_parameter('rpm_scale').value)
        self.max_wheel_rpm = float(self.get_parameter('max_wheel_rpm').value)
        self.encoder_sign = float(self.get_parameter('encoder_sign').value)
        self.cmd_vel_timeout = float(self.get_parameter('cmd_vel_timeout').value)
        
        raw_pub_tf = self.get_parameter('publish_tf').value
        if isinstance(raw_pub_tf, str):
            self.publish_tf = raw_pub_tf.lower() in ('true', '1', 'yes')
        else:
            self.publish_tf = bool(raw_pub_tf)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.wheel_circ = math.pi * self.wheel_d

        # ── Raw Encoder & Outlier Tracking State ──────────────────────
        self._last_raw_us = None
        self._last_fl = None
        self._last_fr = None
        self._last_rl = None
        self._last_rr = None
        self._last_vl = 0.0
        self._last_vr = 0.0

        # ── Robot Odometry State (Dead Reckoning) ─────────────────────
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vth = 0.0
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self.last_time = self.get_clock().now()
        self._last_cmd_time = None          # Thời điểm nhận /cmd_vel gần nhất (watchdog mạng)
        self._cmd_timeout_active = False    # Đã kích hoạt dừng do mất lệnh hay chưa

        # ── Publishers & Subscribers ──────────────────────────────────
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.esp32_cmd_sub = self.create_subscription(
            StringMsg, '/esp32/command', self.esp32_command_callback, 10)
        self.odom_pub = self.create_publisher(
            Odometry, self.odom_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Serial Connection ─────────────────────────────────────────
        self.ser = None
        if self.mode == 'serial':
            self.init_serial()

        self.target_rpm_left = 0.0
        self.target_rpm_right = 0.0
        self.current_rpm_left = 0.0
        self.current_rpm_right = 0.0
        self.max_rpm_accel = 15.0  # RPM/s — Nội suy vận tốc mượt giữa các chu kỳ Camera (0 -> 6.8 RPM trong ~0.45s), chống gật bánh
        self._serial_rx_buffer = ''

        # ── Loop Timer (20 Hz for Odom processing & continuous ESP32 streaming) ──
        self.timer = self.create_timer(0.05, self.update_loop)

        self.get_logger().info(
            f'ESP32 Bridge started [Mode: {self.mode}] [Port: {self.port}] [Wheel D: {self.wheel_d}m, Base: {self.wheel_base}m, CPR: {self.encoder_cpr:.0f} (PPR:{self.encoder_ppr} x Quad:{self.quadrature} x Gear:{self.gear_ratio})]')

    def init_serial(self):
        if not SERIAL_AVAILABLE:
            self.get_logger().warn('pyserial not found! Falling back to MOCK mode.')
            self.mode = 'mock'
            return

        candidate_ports = [self.port, '/dev/esp32', '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB1', '/dev/ttyUSB0']
        seen = set()
        ports_to_try = [p for p in candidate_ports if p and (p not in seen and not seen.add(p))]

        for p in ports_to_try:
            if not os.path.exists(p):
                continue
            try:
                self.ser = serial.Serial(p, self.baud, timeout=0.05)
                self.port = p
                self.get_logger().info(f'✅ Successfully connected to ESP32 on {self.port} at {self.baud} baud')
                return
            except Exception as e:
                self.get_logger().warn(f'Could not open serial port {p}: {e}')

        self.get_logger().warn(f'Could not open any serial port in {ports_to_try}. Running in MOCK mode.')
        self.mode = 'mock'

    def esp32_command_callback(self, msg: StringMsg):
        """
        Sends arbitrary string commands to ESP32 (e.g., RESET_ODOM, PID 1, PID 0, CALIB 1.05).
        """
        cmd = msg.data.strip()
        if not cmd:
            return
        if self.mode == 'serial' and self.ser and self.ser.is_open:
            try:
                self.ser.write(f'{cmd}\n'.encode('utf-8'))
                self.get_logger().info(f'Sent raw command to ESP32: {cmd}')
            except Exception as e:
                self.get_logger().warn(f'Failed to send command to ESP32: {e}')

    def cmd_vel_callback(self, msg: Twist):
        """
        Receives ROS 2 /cmd_vel (linear.x [m/s], angular.z [rad/s]).
        Stores target velocities for closed-loop torque processing in update_loop.
        """
        self._cmd_v = msg.linear.x
        self._cmd_w = msg.angular.z
        self._last_cmd_time = self.get_clock().now()
        if self._cmd_timeout_active:
            self._cmd_timeout_active = False
            self.get_logger().info('✅ [CMD_VEL WATCHDOG] Đã nhận lại lệnh /cmd_vel -> tiếp tục điều khiển.')
        # Chỉ cập nhật vx/vth trực tiếp từ cmd_vel khi chạy chế độ MOCK (không có phần cứng)
        if self.mode != 'serial':
            self.vx = msg.linear.x
            self.vth = msg.angular.z

    def update_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0:
            return

        # ── WATCHDOG MẠNG: mất /cmd_vel (rớt WiFi, laptop treo, CNN crash) -> dừng xe êm ──
        if self._last_cmd_time is not None and self.cmd_vel_timeout > 0.0:
            silence = (now - self._last_cmd_time).nanoseconds / 1e9
            if silence > self.cmd_vel_timeout:
                if not self._cmd_timeout_active:
                    self._cmd_timeout_active = True
                    self.get_logger().warn(
                        f'🛑 [CMD_VEL WATCHDOG] Mất lệnh /cmd_vel {silence:.2f}s > {self.cmd_vel_timeout:.2f}s '
                        f'(rớt mạng / laptop treo?) -> Dừng xe an toàn!')
                self._cmd_v = 0.0
                self._cmd_w = 0.0

        # ── TÍNH TOÁN RPM TRỢ LỰC KÉO TẢI NẶNG (KICKSTART + CLOSED-LOOP TRIM) ──
        v = self._cmd_v
        w = self._cmd_w

        if abs(v) < 0.005 and abs(w) < 0.005:
            self.target_rpm_left = 0.0
            self.target_rpm_right = 0.0
        elif abs(v) < 0.025:
            # Xoay tại chỗ 4 bánh ngược chiều đối xứng (Pivot turn):
            # Cả 4 bánh (2 Trái + 2 Phải) đều quay đồng bộ 11.5 - 15.5 RPM (tốc độ vừa khít độ bám cỏ, kết hợp mô-men xoắn cao trên ESP32)
            rpm_scale_eff = getattr(self, 'rpm_scale', 1.00)
            max_rpm_cap = min(16.0, float(getattr(self, 'max_wheel_rpm', 20.0)))
            pivot_rpm = min(max_rpm_cap, max(11.5, abs(w * self.wheel_base / 2.0 * 60.0 / self.wheel_circ) * max(1.45, rpm_scale_eff)))
            if w > 0:
                rpm_l = -pivot_rpm
                rpm_r = pivot_rpm
            else:
                rpm_l = pivot_rpm
                rpm_r = -pivot_rpm
            self.target_rpm_left = rpm_l
            self.target_rpm_right = rpm_r
        else:
            # Lăn bánh tiến/lùi có bẻ lái (Tracking): Chuẩn tốc độ 0.07 m/s khỏe dư lực vượt mô đất nhỏ (~6.8 RPM)
            v_left = v - (w * self.wheel_base / 2.0)
            v_right = v + (w * self.wheel_base / 2.0)
            rpm_scale_eff = getattr(self, 'rpm_scale', 1.00)
            max_rpm_cap = getattr(self, 'max_wheel_rpm', 15.0)
            rpm_l = ((v_left * 60.0) / self.wheel_circ) * rpm_scale_eff
            rpm_r = ((v_right * 60.0) / self.wheel_circ) * rpm_scale_eff

            min_rpm = float(getattr(self, 'min_moving_rpm', 6.8))
            avg_rpm = (rpm_l + rpm_r) / 2.0
            diff_rpm = (rpm_r - rpm_l) / 2.0
            sign_fwd = 1.0 if avg_rpm >= 0 else -1.0
            eff_fwd = sign_fwd * max(min_rpm, abs(avg_rpm))
            rpm_l = eff_fwd - diff_rpm
            rpm_r = eff_fwd + diff_rpm

            # Khi đang chạy tiến bám luống (sign_fwd > 0 và góc bẻ lái vừa phải):
            # Giữ bánh trong luôn quay tiến tối thiểu >= 4.5 RPM để không bị lì/chết bánh trên cỏ gồ ghề,
            # đồng thời dồn chênh lệch lái sang bánh ngoài để cả 4 bánh luôn chủ động kéo xe tiến!
            if sign_fwd > 0 and abs(w) < 0.35:
                min_inner_rpm = 4.5
                if rpm_l < min_inner_rpm and rpm_r > rpm_l:
                    deficit = min_inner_rpm - rpm_l
                    rpm_l = min_inner_rpm
                    rpm_r = min(max_rpm_cap, rpm_r + deficit)
                elif rpm_r < min_inner_rpm and rpm_l > rpm_r:
                    deficit = min_inner_rpm - rpm_r
                    rpm_r = min_inner_rpm
                    rpm_l = min(max_rpm_cap, rpm_l + deficit)

            # Khóa trần vận tốc từng bánh <= max_wheel_rpm (15.0 RPM)
            max_abs = max(abs(rpm_l), abs(rpm_r))
            if max_abs > max_rpm_cap:
                scale_down = max_rpm_cap / max_abs
                rpm_l *= scale_down
                rpm_r *= scale_down

            self.target_rpm_left = rpm_l
            self.target_rpm_right = rpm_r

        # Slew-Rate Limiter (Bộ làm mềm gia tốc/hãm phanh chống trượt bánh và văng xe trên cỏ/đất)
        # Khi xoay tại chỗ (is_pivot): cấp đối xứng tức thì cho 2 vế Trái/Phải ngược dấu nhau để ESP32 kích hoạt ngay chế độ Pivot Torque,
        # tránh hiện tượng 1 vế đi qua điểm 0 bị khựng/khóa bánh!
        is_pivot = (abs(self._cmd_v) < 0.025 and abs(self._cmd_w) > 0.05)
        if is_pivot:
            self.current_rpm_left = self.target_rpm_left
            self.current_rpm_right = self.target_rpm_right
        else:
            max_delta = self.max_rpm_accel * dt
            diff_l = self.target_rpm_left - self.current_rpm_left
            diff_r = self.target_rpm_right - self.current_rpm_right
            self.current_rpm_left += float(max(-max_delta, min(max_delta, diff_l)))
            self.current_rpm_right += float(max(-max_delta, min(max_delta, diff_r)))

        # Nếu cả target và current đều rất nhỏ (< 0.2 RPM), đưa về 0 để ngắt dứt điểm
        if abs(self.target_rpm_left) < 0.01 and abs(self.current_rpm_left) < 0.2:
            self.current_rpm_left = 0.0
        if abs(self.target_rpm_right) < 0.01 and abs(self.current_rpm_right) < 0.2:
            self.current_rpm_right = 0.0

        # Phát liên tục 20 Hz duy trì lệnh nuôi Watchdog ESP32 mượt mà
        if self.mode == 'serial' and self.ser and self.ser.is_open:
            cmd_str = f'V {self.current_rpm_left:.1f} {self.current_rpm_right:.1f}\n'
            try:
                self.ser.write(cmd_str.encode('utf-8'))
            except Exception as e:
                pass

        # Read serial feedback non-blocking
        odom_received_this_tick = False
        if self.mode == 'serial' and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    raw_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    self._serial_rx_buffer += raw_data

                    while '\n' in self._serial_rx_buffer:
                        line, self._serial_rx_buffer = self._serial_rx_buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue

                        # 1. Giao thức RAW: "RAW <sequence> <timestamp_us> <FL> <FR> <RL> <RR>" (Tính Odometry trực tiếp trên Pi)
                        if line.startswith('RAW'):
                            parts = line.split()
                            if len(parts) >= 7:
                                try:
                                    t_us = int(parts[2])
                                    fl = int(parts[3])
                                    fr = int(parts[4])
                                    rl = int(parts[5])
                                    rr = int(parts[6])
                                except ValueError:
                                    parts = []
                            elif len(parts) >= 6:
                                try:
                                    t_us = int(parts[1])
                                    fl = int(parts[2])
                                    fr = int(parts[3])
                                    rl = int(parts[4])
                                    rr = int(parts[5])
                                except ValueError:
                                    parts = []

                            if len(parts) >= 6:
                                try:

                                    if self._last_raw_us is not None:
                                        # Tính khoảng thời gian delta time tính bằng giây (xử lý tràn micros 32-bit)
                                        dt_us = t_us - self._last_raw_us
                                        if dt_us < 0:
                                            dt_us += (1 << 32)
                                        dt_s = dt_us / 1e6

                                        # Chỉ tính toán nếu chu kỳ delta time hợp lý (5ms đến 250ms)
                                        if 0.005 < dt_s < 0.25:
                                            d_fl = fl - self._last_fl
                                            d_rl = rl - self._last_rl
                                            d_fr = fr - self._last_fr
                                            d_rr = rr - self._last_rr

                                            # Đổi xung vế Trái và vế Phải (có hệ số encoder_sign để khớp hướng chuyển động thực tế)
                                            # Chống sai odom do trượt bánh: nếu 2 bánh cùng vế lệch nhau > 35%,
                                            # bánh quay nhiều hơn đang trượt -> lấy bánh ít xung hơn (bánh đang bám đất).
                                            d_left_ticks = self.encoder_sign * self._side_ticks(d_fl, d_rl)
                                            d_right_ticks = self.encoder_sign * self._side_ticks(d_fr, d_rr)

                                            # Vận tốc tức thời (m/s) = (xung * chu vi bánh / CPR) / dt
                                            v_l_meas = (d_left_ticks * self.wheel_circ / self.encoder_cpr) / dt_s
                                            v_r_meas = (d_right_ticks * self.wheel_circ / self.encoder_cpr) / dt_s

                                            # Lọc làm mượt vận tốc chống sốc dữ liệu mà không vứt bỏ xung
                                            if abs(v_l_meas) < 2.5 and abs(v_r_meas) < 2.5:
                                                if getattr(self, '_has_first_meas', False):
                                                    max_dv = 12.0 * dt_s
                                                    v_l = float(max(self._last_vl - max_dv, min(self._last_vl + max_dv, v_l_meas)))
                                                    v_r = float(max(self._last_vr - max_dv, min(self._last_vr + max_dv, v_r_meas)))
                                                else:
                                                    self._has_first_meas = True
                                                    v_l = v_l_meas
                                                    v_r = v_r_meas
                                                self._last_vl = v_l
                                                self._last_vr = v_r
                                                self.vx = (v_r + v_l) / 2.0
                                                self.vth = (v_r - v_l) / self.wheel_base

                                                # Giám sát đồng bộ hướng: Nếu robot đang được lệnh tiến mà vx bị âm, tự động đồng bộ chuẩn
                                                if self._cmd_v > 0.02 and self.vx < -0.01:
                                                    self.vx = abs(self.vx)
                                                elif self._cmd_v < -0.02 and self.vx > 0.01:
                                                    self.vx = -abs(self.vx)

                                    self._last_raw_us = t_us
                                    self._last_fl = fl
                                    self._last_fr = fr
                                    self._last_rl = rl
                                    self._last_rr = rr
                                except ValueError:
                                    pass

                        # 2. Giao thức ODOM dự phòng: "ODOM <v_left> <v_right>" (chỉ kích hoạt nếu chưa có gói RAW)
                        elif (line.startswith('ODOM') or line.startswith('O ')) and self._last_raw_us is None:
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    v_l = float(parts[1])
                                    v_r = float(parts[2])
                                    self.vx = (v_r + v_l) / 2.0
                                    self.vth = (v_r - v_l) / self.wheel_base
                                    odom_received_this_tick = True
                                except ValueError:
                                    pass

            except Exception:
                pass

        # Dead reckoning integration
        delta_x = (self.vx * math.cos(self.yaw)) * dt
        delta_y = (self.vx * math.sin(self.yaw)) * dt
        delta_th = self.vth * dt

        self.x += delta_x
        self.y += delta_y
        self.yaw += delta_th

        # Normalize yaw [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        # Publish Odometry & TF
        self.publish_odometry(now)

    def _side_ticks(self, d_front, d_rear):
        """Xung đại diện cho 1 vế bánh, loại bỏ bánh đang trượt (quay trơn nhiều xung hơn)."""
        a, b = abs(d_front), abs(d_rear)
        big = max(a, b)
        if big >= 4 and (big - min(a, b)) > 0.35 * big:
            self._slip_count = getattr(self, '_slip_count', 0) + 1
            if self._slip_count % 20 == 1:
                self.get_logger().warn(
                    f'⚠️ [TRƯỢT BÁNH] Vế bánh lệch xung {d_front} vs {d_rear} -> dùng bánh bám đất cho odometry')
            return float(d_front if a < b else d_rear)
        return (d_front + d_rear) / 2.0

    def publish_odometry(self, now):
        # Quaternion from yaw
        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)

        # 1. Publish Odometry message kèm Covariance cho EKF
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance = [
            0.01, 0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.01, 0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  1e6, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 1e6, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 1e6, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.05
        ]

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.vth

        odom.twist.covariance = [
            0.02, 0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  1e6,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  1e6, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 1e6, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 1e6, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.05
        ]

        self.odom_pub.publish(odom)

        # 2. Broadcast TF (odom -> base_footprint)
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


    def destroy_node(self):
        """Clean shutdown: brake motors to 0 and safely close serial port on Ctrl+C."""
        try:
            if self.mode == 'serial' and self.ser and self.ser.is_open:
                for _ in range(2):
                    self.ser.write(b'V 0.0 0.0\n')
                    time.sleep(0.01)
                self.ser.write(b'STOP\n')
                self.ser.flush()
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ESP32Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
