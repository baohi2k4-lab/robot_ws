#!/usr/bin/env python3
"""
==============================================================================
  ROS 2 Node: gps_driver (Package: my_robot_controller)
  
  Trình điều khiển GPS phần cứng tích hợp BỘ LỌC TIỀN XỬ LÝ 4 LỚP:
    Lớp 1: Lọc phẩm chất vệ tinh & HDOP (Bác bỏ khi Sats < 5 hoặc HDOP > 2.5)
    Lớp 2: Lọc ngoại lai động học (Kinematic Spike Rejection - chống nhảy bước do tán cây)
    Lớp 3: Lọc trung vị trượt (Moving Window Median 3-5 mẫu - làm phẳng gai nhọn)
    Lớp 4: Khóa chống trôi khi xe dừng đỗ (Stationary Lock kết hợp /wheel/odom)
==============================================================================
"""

import math
import time
from collections import deque
import statistics
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def parse_nmea_coord(coord_str, direction):
    """Chuyển đổi chuỗi tọa độ NMEA (ddmm.mmmm) sang độ thập phân (Decimal Degrees)."""
    if not coord_str or not direction:
        return float('nan')
    try:
        dot_idx = coord_str.find('.')
        if dot_idx == -1:
            return float('nan')
        deg_len = dot_idx - 2
        deg = float(coord_str[:deg_len])
        minutes = float(coord_str[deg_len:])
        dec_deg = deg + (minutes / 60.0)
        if direction in ['S', 'W']:
            dec_deg = -dec_deg
        return dec_deg
    except Exception:
        return float('nan')


def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách thực tế giữa 2 điểm GPS theo công thức Haversine (đơn vị: Mét)."""
    R = 6371000.0  # Bán kính Trái Đất (m)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class GpsDriverNode(Node):
    def __init__(self):
        super().__init__('gps_driver_node')

        # ── 1. Khai báo tham số cấu hình ──────────────────────────────────
        self.declare_parameter('serial_port', '/dev/ttyAMA0')
        self.declare_parameter('baudrate', 38400)
        self.declare_parameter('frame_id', 'gps_link')
        self.declare_parameter('publish_topic', '/gps/fix')
        self.declare_parameter('vel_publish_topic', '/gps/vel')

        # Tham số bộ lọc 4 lớp
        self.declare_parameter('min_satellites', 5)              # Lớp 1: Tối thiểu 5 vệ tinh
        self.declare_parameter('max_hdop', 2.5)                  # Lớp 1: Ngưỡng HDOP tối đa
        self.declare_parameter('max_velocity_jump', 2.5)         # Lớp 2: Vận tốc nhảy tối đa (m/s)
        self.declare_parameter('max_distance_jump', 2.0)         # Lớp 2: Khoảng cách nhảy tối đa giữa 2 mẫu (m)
        self.declare_parameter('median_window_size', 3)          # Lớp 3: Cửa sổ lọc trung vị (3 mẫu)
        self.declare_parameter('stationary_lock_enabled', True)  # Lớp 4: Bật khóa trôi khi dừng đỗ
        self.declare_parameter('stationary_speed_threshold', 0.03) # Lớp 4: Ngưỡng tốc độ coi là đứng yên (m/s)

        self.port_name = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_topic = self.get_parameter('publish_topic').value
        self.vel_publish_topic = self.get_parameter('vel_publish_topic').value

        self.min_satellites = int(self.get_parameter('min_satellites').value)
        self.max_hdop = float(self.get_parameter('max_hdop').value)
        self.max_velocity_jump = float(self.get_parameter('max_velocity_jump').value)
        self.max_distance_jump = float(self.get_parameter('max_distance_jump').value)
        self.median_window_size = int(self.get_parameter('median_window_size').value)
        self.stationary_lock_enabled = bool(self.get_parameter('stationary_lock_enabled').value)
        self.stationary_speed_threshold = float(self.get_parameter('stationary_speed_threshold').value)

        # ── 2. Bộ nhớ đệm bộ lọc ──────────────────────────────────────────
        self.lat_window = deque(maxlen=self.median_window_size)
        self.lon_window = deque(maxlen=self.median_window_size)
        self.alt_window = deque(maxlen=self.median_window_size)

        self.last_valid_lat = None
        self.last_valid_lon = None
        self.last_valid_time = None
        self.outlier_count = 0

        self.is_robot_moving = False
        self.stationary_lat = None
        self.stationary_lon = None
        self.last_doppler_speed = 0.0
        self.last_cog_deg = 0.0

        # ── 3. Publishers & Subscribers ───────────────────────────────────
        self.publisher_ = self.create_publisher(NavSatFix, self.publish_topic, 10)
        self.vel_publisher_ = self.create_publisher(TwistStamped, self.vel_publish_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, '/wheel/odom', self.odom_callback, 10)

        # ── 4. Kết nối Serial Hardware ────────────────────────────────────
        self.ser = None
        if not SERIAL_AVAILABLE:
            self.get_logger().error("Thư viện pyserial chưa được cài đặt! Hãy chạy: pip install pyserial")
        else:
            try:
                self.ser = serial.Serial(self.port_name, self.baudrate, timeout=1.0)
                self.get_logger().info(f"✅ Đã kết nối thành công GPS Hardware: {self.port_name} @ {self.baudrate} baud.")
            except Exception as e:
                self.get_logger().warn(f"Chưa thể mở cổng {self.port_name}: {e}. (Sẽ tự động thử lại)")

        # Timer chu kỳ đọc Serial 10 Hz
        self.timer = self.create_timer(0.1, self.read_gps_data)

        self.get_logger().info(
            f"🛡️ [GPS Filter] Đã kích hoạt Bộ Lọc 4 Lớp: "
            f"[Min Sats: {self.min_satellites} | Max HDOP: {self.max_hdop:.1f} | "
            f"Max Jump: {self.max_distance_jump:.1f}m | Median: {self.median_window_size} mẫu | "
            f"Stationary Lock: {self.stationary_lock_enabled}] | Pub: {self.publish_topic} & {self.vel_publish_topic}"
        )

    def odom_callback(self, msg: Odometry):
        """Theo dõi chuyển động thực của 4 bánh xe để kích hoạt Lớp 4 (Khóa dừng đỗ)."""
        vx = abs(msg.twist.twist.linear.x)
        wz = abs(msg.twist.twist.angular.z)
        self.is_robot_moving = (vx > self.stationary_speed_threshold or wz > 0.05)

    def read_gps_data(self):
        """Đọc và phân tách bản tin NMEA từ cổng Serial."""
        if not SERIAL_AVAILABLE:
            return

        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial(self.port_name, self.baudrate, timeout=1.0)
                self.get_logger().info(f"Đã kết nối lại thành công cổng GPS {self.port_name}.")
            except Exception:
                return

        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    self.parse_gga(line)
                elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    self.parse_rmc(line)
        except Exception as e:
            self.get_logger().error(f"Lỗi đọc dữ liệu Serial GPS: {e}")

    def parse_rmc(self, line):
        """
        Phân tích cú pháp bản tin RMC để trích xuất vận tốc Doppler (Speed Over Ground)
        và góc hướng di chuyển (Course Over Ground):
        $GNRMC,hhmmss.ss,status,lat,lat_dir,lon,lon_dir,speed_knots,track_deg,ddmmyy,mag_var,var_dir,mode*cs
        """
        parts = line.split(',')
        if len(parts) < 9:
            return
        status = parts[2]
        if status != 'A':
            return

        try:
            speed_knots = float(parts[7]) if parts[7] else 0.0
            speed_mps = speed_knots * 0.5144444444444445
        except ValueError:
            speed_mps = 0.0

        try:
            cog_deg = float(parts[8]) if parts[8] else 0.0
        except ValueError:
            cog_deg = 0.0

        self.last_doppler_speed = speed_mps
        self.last_cog_deg = cog_deg

        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = self.frame_id
        vel_msg.twist.linear.x = speed_mps
        vel_msg.twist.linear.y = 0.0
        vel_msg.twist.linear.z = 0.0
        vel_msg.twist.angular.x = 0.0
        vel_msg.twist.angular.y = 0.0
        vel_msg.twist.angular.z = math.radians(cog_deg)
        self.vel_publisher_.publish(vel_msg)

    def parse_gga(self, line):
        """
        Phân tích cú pháp bản tin GGA và áp dụng Bộ Lọc 4 Lớp:
        $GNGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,qual,sats,hdop,alt,M,geoid,M,dgps_time,dgps_id*cs
        """
        parts = line.split(',')
        if len(parts) < 10:
            return

        raw_lat, lat_dir = parts[2], parts[3]
        raw_lon, lon_dir = parts[4], parts[5]
        fix_quality_str = parts[6]
        num_sats_str = parts[7]
        hdop_str = parts[8]
        raw_alt = parts[9]

        lat = parse_nmea_coord(raw_lat, lat_dir)
        lon = parse_nmea_coord(raw_lon, lon_dir)
        try:
            alt = float(raw_alt) if raw_alt else 0.0
        except ValueError:
            alt = 0.0

        now_time = time.time()
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # ── KIỂM TRA TỌA ĐỘ HỢP LỆ CƠ BẢN ────────────────────────────────
        if math.isnan(lat) or math.isnan(lon):
            self._publish_no_fix(msg)
            return

        try:
            qual = int(fix_quality_str)
        except ValueError:
            qual = 0

        try:
            sats = int(num_sats_str)
        except ValueError:
            sats = 0

        try:
            hdop = float(hdop_str)
        except ValueError:
            hdop = 99.0

        # ── LỚP 1: LỌC PHẨM CHẤT TÍN HIỆU & VỆ TINH ───────────────────────
        if qual == 0 or sats < self.min_satellites or hdop > self.max_hdop:
            self.get_logger().debug(
                f"[Lớp 1 - Bác bỏ] Phẩm chất kém: Qual={qual}, Sats={sats} (<{self.min_satellites}), "
                f"HDOP={hdop:.2f} (>{self.max_hdop:.1f})"
            )
            self._publish_no_fix(msg)
            return

        # ── LỚP 2: LỌC NGOẠI LAI ĐỘNG HỌC (KINEMATIC JUMP REJECTION) ──────
        if self.last_valid_lat is not None and self.last_valid_lon is not None and self.last_valid_time is not None:
            dt = now_time - self.last_valid_time
            if dt > 0.0:
                dist_jump = haversine_distance(self.last_valid_lat, self.last_valid_lon, lat, lon)
                v_apparent = dist_jump / max(dt, 0.05)

                if dist_jump > self.max_distance_jump and v_apparent > self.max_velocity_jump:
                    self.outlier_count += 1
                    if self.outlier_count < 4:
                        self.get_logger().warn(
                            f"🛡️ [Lớp 2 - Outlier Spike] Phát hiện GPS nhảy ảo do phản xạ: "
                            f"Δd = {dist_jump:.2f}m trong {dt:.2f}s (v_ảo = {v_apparent:.1f}m/s) -> ĐÃ BÁC BỎ!"
                        )
                        # Giữ nguyên tọa độ hợp lệ cũ thay vì nhận điểm ngoại lai
                        return
                    else:
                        # Nếu nhảy liên tục > 4 lần, có thể xe bị bốc nhấc di dời vị trí thật -> reset neo
                        self.get_logger().info("🔄 Đã thiết lập lại vị trí neo sau 4 lần nhận diện dịch chuyển lớn.")
                        self.outlier_count = 0
                else:
                    self.outlier_count = 0

        # ── LỚP 3: BỘ LỌC TRUNG VỊ TRƯỢT (MOVING WINDOW MEDIAN) ───────────
        self.lat_window.append(lat)
        self.lon_window.append(lon)
        self.alt_window.append(alt)

        if len(self.lat_window) >= 3:
            filt_lat = statistics.median(self.lat_window)
            filt_lon = statistics.median(self.lon_window)
            filt_alt = statistics.median(self.alt_window)
        else:
            filt_lat = lat
            filt_lon = lon
            filt_alt = alt

        self.last_valid_lat = filt_lat
        self.last_valid_lon = filt_lon
        self.last_valid_time = now_time

        # ── LỚP 4: KHÓA CHỐNG TRÔI KHI DỪNG ĐỖ (ZERO-VELOCITY LOCK) ───────
        out_lat = filt_lat
        out_lon = filt_lon

        if self.stationary_lock_enabled and not self.is_robot_moving:
            if self.stationary_lat is None:
                self.stationary_lat = filt_lat
                self.stationary_lon = filt_lon
            else:
                drift_dist = haversine_distance(self.stationary_lat, self.stationary_lon, filt_lat, filt_lon)
                if drift_dist < 0.8:
                    # Ghim chặt tọa độ tại điểm neo dừng, triệt tiêu 100% rung lắc tại chỗ
                    out_lat = self.stationary_lat
                    out_lon = self.stationary_lon
                else:
                    # Nếu trôi xa hơn 0.8m, cập nhật lại điểm neo
                    self.stationary_lat = filt_lat
                    self.stationary_lon = filt_lon
        else:
            # Xe đang chạy -> Xóa cờ neo dừng đỗ
            self.stationary_lat = None
            self.stationary_lon = None

        # ── XUẤT DỮ LIỆU ĐÃ LỌC SẠCH LÊN TOPIC /gps/fix ────────────────────
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.latitude = out_lat
        msg.longitude = out_lon
        msg.altitude = filt_alt

        # Tính ma trận hiệp phương sai sai số vị trí từ HDOP
        var_h = max(0.4, (hdop * 1.2)) ** 2
        var_v = max(0.8, (hdop * 2.5)) ** 2

        # Nếu đang khóa dừng đỗ, tăng độ tin cậy của vị trí (giảm phương sai)
        if self.stationary_lock_enabled and not self.is_robot_moving and self.stationary_lat is not None:
            var_h = 0.15

        msg.position_covariance = [
            var_h, 0.0, 0.0,
            0.0, var_h, 0.0,
            0.0, 0.0, var_v
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        self.publisher_.publish(msg)
        self.get_logger().debug(
            f"[GPS Filtered] Lat={out_lat:.8f}, Lon={out_lon:.8f}, Sats={sats}, HDOP={hdop:.2f}, Var={var_h:.2f}"
        )

    def _publish_no_fix(self, msg):
        """Phát gói tin NO FIX khi chưa khóa vệ tinh hoặc tín hiệu bị suy giảm."""
        msg.status.status = NavSatStatus.STATUS_NO_FIX
        msg.latitude = float('nan')
        msg.longitude = float('nan')
        msg.altitude = float('nan')
        msg.position_covariance = [0.0] * 9
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GpsDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
