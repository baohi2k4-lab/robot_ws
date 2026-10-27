#!/usr/bin/env python3
"""
Node điều khiển motor BTS7960 qua Raspberry Pi GPIO.

Sơ đồ nối dây (mỗi bên xe cần 1 module BTS7960):
  BTS7960 Left Motor:
    RPWM → GPIO pin số 'left_rpwm_pin'   (quay tiến)
    LPWM → GPIO pin số 'left_lpwm_pin'   (quay lùi)
    R_EN → GPIO pin số 'left_en_pin'     (enable HIGH)
    L_EN → GPIO pin số 'left_en_pin'     (enable HIGH, nối chung với R_EN)

  BTS7960 Right Motor:
    RPWM → GPIO pin số 'right_rpwm_pin'
    LPWM → GPIO pin số 'right_lpwm_pin'
    R_EN → GPIO pin số 'right_en_pin'
    L_EN → GPIO pin số 'right_en_pin'

Cài đặt trước khi dùng:
  sudo apt install python3-rpi.gpio
  pip3 install RPi.GPIO
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False


class BTS7960DriverNode(Node):
    """
    Nhận lệnh /cmd_vel (Twist) và điều khiển 2 motor qua BTS7960.

    Differential drive kinematics:
        v_left  = linear_x - angular_z * wheel_base / 2
        v_right = linear_x + angular_z * wheel_base / 2
    """

    def __init__(self):
        super().__init__('bts7960_driver_node')
        self.get_logger().info('Initializing BTS7960 motor driver node...')

        # ── Robot physical parameters ────────────────────────────────────
        self.declare_parameter('wheel_base', 0.58)       # m  (chassis 0.53 + 2×wheel_ygap)
        self.declare_parameter('max_linear_speed', 0.16) # m/s → 100% PWM (ở 0.07 m/s đạt ~44% PWM, tạo xung khỏe vượt cản)
        self.declare_parameter('pwm_frequency', 1500)    # Hz — Tần số xung tối ưu cho motor 775 có hộp số 220 RPM
        self.declare_parameter('min_duty_cycle', 38.0)   # % — Sàn 38% PWM để motor 775 luôn đủ mô-men xoắn thắng ma sát tĩnh

        # ── GPIO pin numbers (BCM numbering) ─────────────────────────────
        self.declare_parameter('left_rpwm_pin', 17)   # trái tiến
        self.declare_parameter('left_lpwm_pin', 27)   # trái lùi
        self.declare_parameter('left_en_pin',   22)   # enable trái
        self.declare_parameter('right_rpwm_pin', 23)  # phải tiến
        self.declare_parameter('right_lpwm_pin', 24)  # phải lùi
        self.declare_parameter('right_en_pin',   25)  # enable phải

        self.wheel_base       = self.get_parameter('wheel_base').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.pwm_freq         = self.get_parameter('pwm_frequency').value
        self.min_duty_cycle   = self.get_parameter('min_duty_cycle').value

        self.left_rpwm_pin  = self.get_parameter('left_rpwm_pin').value
        self.left_lpwm_pin  = self.get_parameter('left_lpwm_pin').value
        self.left_en_pin    = self.get_parameter('left_en_pin').value
        self.right_rpwm_pin = self.get_parameter('right_rpwm_pin').value
        self.right_lpwm_pin = self.get_parameter('right_lpwm_pin').value
        self.right_en_pin   = self.get_parameter('right_en_pin').value

        # ── GPIO setup ───────────────────────────────────────────────────
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            for pin in [self.left_rpwm_pin, self.left_lpwm_pin, self.left_en_pin,
                        self.right_rpwm_pin, self.right_lpwm_pin, self.right_en_pin]:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

            # Enable pins HIGH → BTS7960 active
            GPIO.output(self.left_en_pin,  GPIO.HIGH)
            GPIO.output(self.right_en_pin, GPIO.HIGH)

            # PWM objects
            self._left_rpwm  = GPIO.PWM(self.left_rpwm_pin,  self.pwm_freq)
            self._left_lpwm  = GPIO.PWM(self.left_lpwm_pin,  self.pwm_freq)
            self._right_rpwm = GPIO.PWM(self.right_rpwm_pin, self.pwm_freq)
            self._right_lpwm = GPIO.PWM(self.right_lpwm_pin, self.pwm_freq)

            for pwm in [self._left_rpwm, self._left_lpwm,
                        self._right_rpwm, self._right_lpwm]:
                pwm.start(0)

            self.get_logger().info('GPIO initialized successfully.')
        else:
            self.get_logger().warn(
                'RPi.GPIO not available — running in DRY-RUN mode (no actual motor output).'
            )

        # ── ROS subscriber ───────────────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_callback, 10
        )

        # Safety: dừng motor nếu không nhận được lệnh trong 0.5 giây
        self._watchdog = self.create_timer(0.5, self._watchdog_callback)
        self._last_cmd_time = self.get_clock().now()

        self.get_logger().info('BTS7960 driver ready. Listening to /cmd_vel ...')

    # ────────────────────────────────────────────────────────────────────
    def _cmd_vel_callback(self, msg: Twist):
        self._last_cmd_time = self.get_clock().now()

        linear_x  = msg.linear.x
        angular_z = msg.angular.z

        # Differential drive: tính vận tốc dài mỗi bánh (m/s)
        v_left  = linear_x - angular_z * self.wheel_base / 2.0
        v_right = linear_x + angular_z * self.wheel_base / 2.0

        # Smooth Forward-Bias Kinematics:
        # Khi xe đang chạy tiến (linear_x > 0.02 m/s), cả 2 bánh phải luôn duy trì lực kéo tiến ổn định.
        # Tuyệt đối không để bánh trong bị đảo chiều giật lùi gây khựng xe và mất lực.
        # Bánh trong giữ vận tốc tiến sàn tối thiểu (min_fwd), bánh ngoài tăng tốc tương ứng để tạo đúng lực quay.
        if linear_x > 0.02:
            min_fwd = 0.055  # m/s (~38% duty sàn để motor 775 220RPM luôn có mô-men đẩy tiến đầm chắc)
            min_v = min(v_left, v_right)
            if min_v < min_fwd:
                shift = min_fwd - min_v
                v_left += shift
                v_right += shift

        # Chuyển sang duty cycle 0–100%
        duty_left  = self._vel_to_duty(v_left)
        duty_right = self._vel_to_duty(v_right)

        self._set_motor('left',  duty_left)
        self._set_motor('right', duty_right)

        self.get_logger().info(
            f'[MOTOR] L={duty_left:+.1f}%  R={duty_right:+.1f}%  '
            f'(lin={linear_x:.2f} ang={angular_z:.2f})',
            throttle_duration_sec=0.5
        )

    def _watchdog_callback(self):
        """Dừng xe nếu không nhận lệnh mới trong 0.5 giây."""
        dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if dt > 0.5:
            self._set_motor('left',  0.0)
            self._set_motor('right', 0.0)

    # ────────────────────────────────────────────────────────────────────
    def _vel_to_duty(self, velocity_ms: float) -> float:
        """Chuyển vận tốc (m/s) → duty cycle có dấu (−100 .. +100).
        Áp dụng ngưỡng sàn min_duty_cycle để motor có đủ momen khởi động
        trên nền cỏ/đất, tránh bị kẹt ở PWM quá thấp."""
        clamped = max(-self.max_linear_speed,
                      min(self.max_linear_speed, velocity_ms))
        raw_duty = clamped / self.max_linear_speed * 100.0

        # Áp dụng ngưỡng sàn PWM: nếu yêu cầu quay (|duty| > 0.5%)
        # nhưng PWM thấp hơn min_duty_cycle → nâng lên min_duty_cycle
        if abs(raw_duty) > 0.5 and abs(raw_duty) < self.min_duty_cycle:
            raw_duty = self.min_duty_cycle if raw_duty > 0 else -self.min_duty_cycle

        return raw_duty

    def _set_motor(self, side: str, duty: float):
        """
        Điều khiển một motor.
          duty > 0 → tiến  (RPWM = |duty|, LPWM = 0)
          duty < 0 → lùi   (RPWM = 0,     LPWM = |duty|)
          duty = 0 → dừng  (cả hai = 0)
        """
        abs_duty = min(100.0, abs(duty))

        if not GPIO_AVAILABLE:
            return  # dry-run

        if side == 'left':
            rpwm, lpwm = self._left_rpwm, self._left_lpwm
        else:
            rpwm, lpwm = self._right_rpwm, self._right_lpwm

        if duty > 0.5:
            rpwm.ChangeDutyCycle(abs_duty)
            lpwm.ChangeDutyCycle(0)
        elif duty < -0.5:
            rpwm.ChangeDutyCycle(0)
            lpwm.ChangeDutyCycle(abs_duty)
        else:
            rpwm.ChangeDutyCycle(0)
            lpwm.ChangeDutyCycle(0)

    # ────────────────────────────────────────────────────────────────────
    def destroy_node(self):
        """Cleanup GPIO khi tắt node."""
        if GPIO_AVAILABLE:
            for pwm in [self._left_rpwm, self._left_lpwm,
                        self._right_rpwm, self._right_lpwm]:
                pwm.stop()
            GPIO.cleanup()
            self.get_logger().info('GPIO cleaned up.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BTS7960DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Motor driver node interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
