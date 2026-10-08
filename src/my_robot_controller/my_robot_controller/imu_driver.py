#!/usr/bin/env python3
"""
ROS 2 Hardware IMU Driver Node for ICM-20948 (9-DoF I2C Motion Sensor).
Đọc dữ liệu Gia tốc (Accel), Vận tốc góc (Gyro) và Từ kế (Mag) từ module ICM-20948
gắn trên chân I2C của Raspberry Pi (SDA Pin 3, SCL Pin 5).

Tính năng:
1. Giao tiếp I2C Native (hoạt động độc lập không cần thư viện ngoài) hoặc qua smbus2.
2. Tự động chuyển đổi Bank 0 / Bank 2 và cấu hình bộ lọc chống nhiễu DLPF.
3. Tự động cân bằng Zero-Bias Gyroscope khi vừa khởi động.
4. Bộ lọc dung hợp Complementary Orientation Filter tính toán góc nghiêng Roll, Pitch, Yaw
   và chuyển đổi sang Quaternion chuẩn sensor_msgs/msg/Imu.
5. Phát dữ liệu lên topic /imu @ 50 Hz.
"""

import os
import math
import time
import struct
import fcntl
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry


# ── ICM-20948 I2C Registers & Constants ─────────────────────────────────
I2C_SLAVE = 0x0703

# Common Register
REG_BANK_SEL = 0x7F

# Bank 0 Registers
WHO_AM_I     = 0x00
USER_CTRL    = 0x03
PWR_MGMT_1   = 0x06
PWR_MGMT_2   = 0x07
INT_PIN_CFG  = 0x0F
ACCEL_XOUT_H = 0x2D
GYRO_XOUT_H  = 0x33
TEMP_OUT_H   = 0x39

# Bank 2 Registers
GYRO_CONFIG_1 = 0x01
ACCEL_CONFIG  = 0x14

# Conversion Constants
GRAVITY_MSS   = 9.80665
DEG_TO_RAD    = math.pi / 180.0


class I2CInterface:
    """Giao tiếp I2C cấp thấp chuẩn Linux /dev/i2c-X."""
    def __init__(self, bus_num=1, address=0x68):
        self.bus_num = bus_num
        self.address = address
        self.dev_path = f"/dev/i2c-{bus_num}"
        self.fd = None
        self.open()

    def open(self):
        try:
            if os.path.exists(self.dev_path):
                self.fd = os.open(self.dev_path, os.O_RDWR)
                fcntl.ioctl(self.fd, I2C_SLAVE, self.address)
                return True
        except Exception:
            self.fd = None
        return False

    def is_open(self):
        return self.fd is not None

    def write_byte(self, reg, val):
        if self.fd is None:
            return False
        try:
            os.write(self.fd, bytes([reg, val & 0xFF]))
            return True
        except Exception:
            return False

    def read_bytes(self, reg, length):
        if self.fd is None:
            return None
        try:
            os.write(self.fd, bytes([reg]))
            data = os.read(self.fd, length)
            return data if len(data) == length else None
        except Exception:
            return None

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None


class ICM20948Driver:
    """Driver điều khiển cảm biến 9 trục ICM-20948."""
    def __init__(self, i2c: I2CInterface):
        self.i2c = i2c
        self.accel_scale = 8192.0   # ±4g -> 8192 LSB/g
        self.gyro_scale = 65.5      # ±500 dps -> 65.5 LSB/dps
        self.is_initialized = False

    def select_bank(self, bank):
        """Chuyển đổi User Bank (0, 1, 2, 3)."""
        return self.i2c.write_byte(REG_BANK_SEL, (bank & 0x03) << 4)

    def initialize(self):
        if not self.i2c.is_open() and not self.i2c.open():
            return False

        # 1. Reset chip
        self.select_bank(0)
        self.i2c.write_byte(PWR_MGMT_1, 0x80)
        time.sleep(0.05)

        # 2. Wake up & Auto Clock Select
        self.select_bank(0)
        self.i2c.write_byte(PWR_MGMT_1, 0x01)
        time.sleep(0.02)

        # 3. Enable Accelerometer & Gyroscope
        self.i2c.write_byte(PWR_MGMT_2, 0x00)

        # 4. Kiểm tra WHO_AM_I (0xEA hoặc 0x98)
        data = self.i2c.read_bytes(WHO_AM_I, 1)
        if not data or data[0] not in (0xEA, 0x98):
            return False

        # 5. Cấu hình Bank 2: Gyro ±500 dps, DLPF 3
        self.select_bank(2)
        self.i2c.write_byte(GYRO_CONFIG_1, (0x01 << 1) | 0x01)  # ±500 dps, DLPF on

        # 6. Cấu hình Bank 2: Accel ±4g, DLPF 3
        self.i2c.write_byte(ACCEL_CONFIG, (0x01 << 1) | 0x01)   # ±4g, DLPF on

        # 7. Quay lại Bank 0 để đọc dữ liệu
        self.select_bank(0)
        self.is_initialized = True
        return True

    def read_raw_sensors(self):
        """Đọc đồng thời 12 byte Gia tốc và Vận tốc góc từ Bank 0."""
        if not self.is_initialized:
            return None

        # Đọc 12 bytes liên tục từ ACCEL_XOUT_H (0x2D) đến GYRO_ZOUT_L (0x38)
        data = self.i2c.read_bytes(ACCEL_XOUT_H, 12)
        if not data or len(data) < 12:
            return None

        # Unpack 6 số nguyên 16-bit có dấu (Big Endian)
        ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw = struct.unpack('>6h', data)

        # Chuyển đổi sang đơn vị chuẩn SI
        # Accel: m/s^2
        ax = (ax_raw / self.accel_scale) * GRAVITY_MSS
        ay = (ay_raw / self.accel_scale) * GRAVITY_MSS
        az = (az_raw / self.accel_scale) * GRAVITY_MSS

        # Gyro: rad/s
        gx = (gx_raw / self.gyro_scale) * DEG_TO_RAD
        gy = (gy_raw / self.gyro_scale) * DEG_TO_RAD
        gz = (gz_raw / self.gyro_scale) * DEG_TO_RAD

        return ax, ay, az, gx, gy, gz


class ImuDriverNode(Node):
    def __init__(self):
        super().__init__('imu_driver_node')

        # Parameters
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x68)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_topic', '/imu/data')
        self.declare_parameter('raw_topic', '/imu/data_raw')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('calibrate_samples', 60)
        self.declare_parameter('vibration_filter_enabled', True)
        self.declare_parameter('accel_ema_alpha', 0.75)          # 75% mẫu cũ + 25% mẫu mới -> lọc sạch rung cơ học 775
        self.declare_parameter('gyro_ema_alpha', 0.80)           # Làm mịn nhẹ gyro
        self.declare_parameter('adaptive_bias_tracking', True)   # Tự động bám bù trôi nhiệt độ khi xe dừng
        self.declare_parameter('stationary_speed_threshold', 0.02) # Ngưỡng coi xe đang dừng (m/s)
        self.declare_parameter('mount_upside_down', True)        # Module ICM-20948 gắn úp mặt xuống sàn xe (quay 180° trục X)

        self.bus_num = int(self.get_parameter('i2c_bus').value)
        self.address = int(self.get_parameter('i2c_address').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_topic = self.get_parameter('publish_topic').value
        self.raw_topic = self.get_parameter('raw_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.calib_target = int(self.get_parameter('calibrate_samples').value)

        self.vibration_filter_enabled = bool(self.get_parameter('vibration_filter_enabled').value)
        self.accel_ema_alpha = float(self.get_parameter('accel_ema_alpha').value)
        self.gyro_ema_alpha = float(self.get_parameter('gyro_ema_alpha').value)
        self.adaptive_bias_tracking = bool(self.get_parameter('adaptive_bias_tracking').value)
        self.stationary_speed_threshold = float(self.get_parameter('stationary_speed_threshold').value)
        self.mount_upside_down = bool(self.get_parameter('mount_upside_down').value)

        # Publishers & Subscribers
        self.imu_pub = self.create_publisher(Imu, self.publish_topic, 10)
        self.imu_raw_pub = self.create_publisher(Imu, self.raw_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, '/wheel/odom', self.odom_callback, 10)

        # I2C & Driver
        self.i2c = I2CInterface(bus_num=self.bus_num, address=self.address)
        self.sensor = ICM20948Driver(self.i2c)

        # Calibration State
        self.calibrating = True
        self.calib_count = 0
        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0

        # Bộ lọc chống rung động cơ 775 (Low-Pass EMA Filter)
        self.filt_ax = 0.0
        self.filt_ay = 0.0
        self.filt_az = GRAVITY_MSS
        self.filt_gx = 0.0
        self.filt_gy = 0.0
        self.filt_gz = 0.0

        # Trạng thái theo dõi xe dừng đỗ để bám bù trôi nhiệt độ (ZUPT)
        self.is_stationary = True
        self.stationary_start_time = time.time()
        self.zupt_learning_rate = 0.005  # Tốc độ học vi sai Zero-bias (0.5% mỗi mẫu @ 50Hz)

        # Orientation Filter State (Euler -> Quaternion)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha = 0.96  # Complementary filter weight (96% gyro, 4% gravity)
        self.last_time = time.time()
        self.last_log_time = 0.0

        # Connect to Hardware
        if self.sensor.initialize():
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"🚀 [ICM-20948 IMU] ĐÃ KẾT NỐI THÀNH CÔNG!")
            self.get_logger().info(f"📍 I2C: /dev/i2c-{self.bus_num} @ 0x{self.address:02X} | Topic: {self.publish_topic}")
            self.get_logger().info(f"🔄 Đang tự động hiệu chuẩn Zero-Bias Gyroscope ({self.calib_target} mẫu)...")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().warn(
                f"⚠️ Không tìm thấy cảm biến ICM-20948 tại /dev/i2c-{self.bus_num} (0x{self.address:02X}).\n"
                f"   Kiểm tra chân cắm: 3.3V, GND, SDA (Pin 3), SCL (Pin 5) hoặc lệnh 'sudo i2cdetect -y 1'."
            )

        # Timer Loop (50 Hz)
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self.timer_callback)

    def odom_callback(self, msg: Odometry):
        """Theo dõi vận tốc từ encoder 4 bánh để phát hiện xe đứng yên (ZUPT)."""
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        speed = abs(vx)

        now = time.time()
        if speed < self.stationary_speed_threshold and abs(wz) < 0.03:
            if not self.is_stationary:
                self.is_stationary = True
                self.stationary_start_time = now
        else:
            self.is_stationary = False
            self.stationary_start_time = now

    def timer_callback(self):
        raw = self.sensor.read_raw_sensors()
        if raw is None:
            # Thử khởi tạo lại định kỳ nếu mất kết nối
            if not self.sensor.is_initialized:
                self.sensor.initialize()
            return

        ax, ay, az, gx, gy, gz = raw

        # Chuyển đổi hệ quy chiếu cảm biến sang hệ quy chiếu chuẩn Robot ROS REP-103
        # Khi module ICM-20948 được bắt ốc úp mặt xuống sàn xe (quay 180° quanh trục X dọc thân xe):
        # - X_robot =  X_sensor (tiến về phía trước)
        # - Y_robot = -Y_sensor (hướng sang trái)
        # - Z_robot = -Z_sensor (hướng thẳng đứng lên trên trời, trọng lực +9.81 m/s²)
        # - Gyro: g_x = g_x, g_y = -g_y, g_z = -g_z (quay Trái thì Wz > 0 chuẩn CCW)
        if self.mount_upside_down:
            ay = -ay
            az = -az
            gy = -gy
            gz = -gz

        now = time.time()
        dt = max(0.001, min(0.1, now - self.last_time))
        self.last_time = now

        # Giai đoạn 1: Hiệu chuẩn Gyro Zero-Bias khi đứng yên ban đầu
        if self.calibrating:
            self.gyro_bias_x += gx
            self.gyro_bias_y += gy
            self.gyro_bias_z += gz
            self.calib_count += 1

            if self.calib_count >= self.calib_target:
                self.gyro_bias_x /= self.calib_count
                self.gyro_bias_y /= self.calib_count
                self.gyro_bias_z /= self.calib_count
                self.calibrating = False

                # Khởi tạo bộ lọc EMA bằng giá trị cảm biến đầu tiên
                self.filt_ax = ax
                self.filt_ay = ay
                self.filt_az = az
                self.filt_gx = 0.0
                self.filt_gy = 0.0
                self.filt_gz = 0.0

                # Khởi tạo góc nghiêng ban đầu từ gia tốc kế
                self.roll = math.atan2(ay, az)
                self.pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
                self.get_logger().info(
                    f"✅ [ICM-20948] HIỆU CHUẨN XONG! Bias Gyro (deg/s): "
                    f"X={self.gyro_bias_x/DEG_TO_RAD:.2f}, Y={self.gyro_bias_y/DEG_TO_RAD:.2f}, Z={self.gyro_bias_z/DEG_TO_RAD:.2f}"
                )
            return

        # Giai đoạn 2: Trừ Zero Bias
        gx_corr = gx - self.gyro_bias_x
        gy_corr = gy - self.gyro_bias_y
        gz_corr = gz - self.gyro_bias_z

        # Adaptive Zero-Velocity Gyro Bias Tracking (ZUPT)
        # Khi xe dừng đỗ (vận tốc bánh xe = 0 > 1.0s), cập nhật bù trôi nhiệt độ vi sai
        if self.adaptive_bias_tracking and self.is_stationary and (now - self.stationary_start_time) > 1.0:
            self.gyro_bias_x = (1.0 - self.zupt_learning_rate) * self.gyro_bias_x + self.zupt_learning_rate * gx
            self.gyro_bias_y = (1.0 - self.zupt_learning_rate) * self.gyro_bias_y + self.zupt_learning_rate * gy
            self.gyro_bias_z = (1.0 - self.zupt_learning_rate) * self.gyro_bias_z + self.zupt_learning_rate * gz
            gx_corr = gx - self.gyro_bias_x
            gy_corr = gy - self.gyro_bias_y
            gz_corr = gz - self.gyro_bias_z

        # Bộ lọc chống rung động cơ 775 và hộp số (Low-Pass EMA Filter & Spike Clamping)
        # 1. Cắt tỉa sốc cơ học (Spike clamping) quá mức ±25 m/s² (~2.5g)
        ax = max(-25.0, min(25.0, ax))
        ay = max(-25.0, min(25.0, ay))
        az = max(-25.0, min(25.0, az))

        if self.vibration_filter_enabled:
            # Low-Pass Exponential Moving Average (EMA)
            self.filt_ax = self.accel_ema_alpha * self.filt_ax + (1.0 - self.accel_ema_alpha) * ax
            self.filt_ay = self.accel_ema_alpha * self.filt_ay + (1.0 - self.accel_ema_alpha) * ay
            self.filt_az = self.accel_ema_alpha * self.filt_az + (1.0 - self.accel_ema_alpha) * az

            self.filt_gx = self.gyro_ema_alpha * self.filt_gx + (1.0 - self.gyro_ema_alpha) * gx_corr
            self.filt_gy = self.gyro_ema_alpha * self.filt_gy + (1.0 - self.gyro_ema_alpha) * gy_corr
            self.filt_gz = self.gyro_ema_alpha * self.filt_gz + (1.0 - self.gyro_ema_alpha) * gz_corr

            # Triệt tiêu nhiễu tĩnh MEMS khi xe dừng đỗ (Deadband 0.015 rad/s ~ 0.85 deg/s)
            if self.is_stationary and (now - self.stationary_start_time) > 0.5:
                if abs(self.filt_gx) < 0.015: self.filt_gx = 0.0
                if abs(self.filt_gy) < 0.015: self.filt_gy = 0.0
                if abs(self.filt_gz) < 0.015: self.filt_gz = 0.0

            out_ax, out_ay, out_az = self.filt_ax, self.filt_ay, self.filt_az
            out_gx, out_gy, out_gz = self.filt_gx, self.filt_gy, self.filt_gz
        else:
            if self.is_stationary and (now - self.stationary_start_time) > 0.5:
                if abs(gx_corr) < 0.015: gx_corr = 0.0
                if abs(gy_corr) < 0.015: gy_corr = 0.0
                if abs(gz_corr) < 0.015: gz_corr = 0.0
            out_ax, out_ay, out_az = ax, ay, az
            out_gx, out_gy, out_gz = gx_corr, gy_corr, gz_corr

        # Tính góc nghiêng tĩnh từ gia tốc trọng trường đã lọc
        roll_acc = math.atan2(out_ay, out_az)
        pitch_acc = math.atan2(-out_ax, math.sqrt(out_ay * out_ay + out_az * out_az))

        # Bộ lọc dung hợp Complementary Filter
        self.roll = self.alpha * (self.roll + out_gx * dt) + (1.0 - self.alpha) * roll_acc
        self.pitch = self.alpha * (self.pitch + out_gy * dt) + (1.0 - self.alpha) * pitch_acc
        self.yaw += out_gz * dt

        # Chuyển đổi Euler (Roll, Pitch, Yaw) sang Quaternion chuẩn ROS
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cp = math.cos(self.pitch * 0.5)
        sp = math.sin(self.pitch * 0.5)
        cr = math.cos(self.roll * 0.5)
        sr = math.sin(self.roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # 1. Publish Raw IMU message cho bộ lọc ngoài (Madgwick AHRS Filter)
        now_msg = self.get_clock().now().to_msg()
        raw_msg = Imu()
        raw_msg.header.stamp = now_msg
        raw_msg.header.frame_id = self.frame_id
        # Orientation unknown -> orientation_covariance[0] = -1
        raw_msg.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raw_msg.angular_velocity.x = out_gx
        raw_msg.angular_velocity.y = out_gy
        raw_msg.angular_velocity.z = out_gz
        raw_msg.angular_velocity_covariance = [
            0.0001, 0.0, 0.0,
            0.0, 0.0001, 0.0,
            0.0, 0.0, 0.0001
        ]
        raw_msg.linear_acceleration.x = out_ax
        raw_msg.linear_acceleration.y = out_ay
        raw_msg.linear_acceleration.z = out_az
        raw_msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]
        self.imu_raw_pub.publish(raw_msg)

        # 2. Publish Internal Filtered IMU message (Dự phòng Complementary Filter)
        msg = Imu()
        msg.header.stamp = now_msg
        msg.header.frame_id = self.frame_id

        msg.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        msg.orientation_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.005
        ]

        msg.angular_velocity.x = out_gx
        msg.angular_velocity.y = out_gy
        msg.angular_velocity.z = out_gz
        msg.angular_velocity_covariance = [
            0.0001, 0.0, 0.0,
            0.0, 0.0001, 0.0,
            0.0, 0.0, 0.0001
        ]

        msg.linear_acceleration.x = out_ax
        msg.linear_acceleration.y = out_ay
        msg.linear_acceleration.z = out_az
        msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        self.imu_pub.publish(msg)

        # Định kỳ in trạng thái trực quan ra màn hình terminal (1 Hz) để người dùng theo dõi không bị rối
        if (now - self.last_log_time) >= 1.0:
            self.last_log_time = now
            roll_deg = self.roll / DEG_TO_RAD
            pitch_deg = self.pitch / DEG_TO_RAD
            yaw_deg = self.yaw / DEG_TO_RAD
            stat_str = "TĨNH (ZUPT)" if self.is_stationary else "CHẠY"
            self.get_logger().info(
                f"📐 [IMU Live] Roll: {roll_deg:+5.1f}° | Pitch: {pitch_deg:+5.1f}° | Yaw: {yaw_deg:+5.1f}° | "
                f"Acc: ({out_ax:+5.2f}, {out_ay:+5.2f}, {out_az:+5.2f}) | Wz: {out_gz:+5.3f} rad/s | [{stat_str}]"
            )

    def destroy_node(self):
        if self.i2c:
            self.i2c.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
