#!/usr/bin/env python3
"""
BÀN PHÍM ĐIỀU KHIỂN ROBOT (CHẠY LIÊN TỤC KHÔNG NGẮT)
- Hỗ trợ cả 2 chuẩn:
  1. Phím WASD: W (Tiến), S (Lùi), A (Xoay Trái), D (Xoay Phải), Space/X (Dừng)
  2. Phím ROS chuẩn: i, ,, j, l, k, u, o, m, .
- Tự động phát liên tục ~16 Hz nuôi ESP32 Watchdog: Bấm 1 lần là xe chạy liên tục không bao giờ bị ngắt!
- Bấm [Space], [X] hoặc [K] để DỪNG XE.
"""

import os
import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

BANNER = """
================================================================================
          🤖 BÀN PHÍM ĐIỀU KHIỂN ROBOT (CHẠY LIÊN TỤC KHÔNG TỰ DỪNG)
================================================================================
  [ĐIỀU HƯỚNG BẰNG PHÍM WASD HOẶC IJKL]
      [W] hoặc [i] : TIẾN THẲNG
      [S] hoặc [,] : LÙI THẲNG
      [A] hoặc [j] : XOAY TRÁI
      [D] hoặc [l] : XOAY PHẢI
      [Q] hoặc [u] : Tiến Rẽ Trái       [E] hoặc [o] : Tiến Rẽ Phải
      [Z] hoặc [m] : Lùi Rẽ Trái        [C] hoặc [.] : Lùi Rẽ Phải

  👉 PHANH DỪNG XE: [Space], [X] hoặc [K]
  👉 ĐIỀU CHỈNH TỐC ĐỘ:
     [+] hoặc [1] : Tăng tốc độ (+10%)
     [-] hoặc [2] : Giảm tốc độ (-10%)
  👉 THOÁT CHƯƠNG TRÌNH: [Ctrl + C]
================================================================================
"""

MOVE_BINDINGS = {
    # ── Chuẩn WASD ──────────────────────────────
    'w': (1.0, 0.0),
    'W': (1.0, 0.0),
    's': (-1.0, 0.0),
    'S': (-1.0, 0.0),
    'a': (0.0, 1.0),
    'A': (0.0, 1.0),
    'd': (0.0, -1.0),
    'D': (0.0, -1.0),
    'q': (1.0, 1.0),
    'Q': (1.0, 1.0),
    'e': (1.0, -1.0),
    'E': (1.0, -1.0),
    'z': (-1.0, -1.0),
    'Z': (-1.0, -1.0),
    'c': (-1.0, 1.0),
    'C': (-1.0, 1.0),
    'x': (0.0, 0.0),
    'X': (0.0, 0.0),
    # ── Chuẩn ROS 2 gốc (i, ,, j, l, k) ─────────
    'i': (1.0, 0.0),
    'I': (1.0, 0.0),
    ',': (-1.0, 0.0),
    '<': (-1.0, 0.0),
    'j': (0.0, 1.0),
    'J': (0.0, 1.0),
    'l': (0.0, -1.0),
    'L': (0.0, -1.0),
    'u': (1.0, 1.0),
    'U': (1.0, 1.0),
    'o': (1.0, -1.0),
    'O': (1.0, -1.0),
    'm': (-1.0, -1.0),
    'M': (-1.0, -1.0),
    '.': (-1.0, 1.0),
    '>': (-1.0, 1.0),
    'k': (0.0, 0.0),
    'K': (0.0, 0.0),
    ' ': (0.0, 0.0),
}

SPEED_BINDINGS = {
    '+': 1.1,
    '=': 1.1,
    '1': 1.1,
    '-': 0.9,
    '_': 0.9,
    '2': 0.9,
}


def get_key(settings, timeout=0.05):
    if settings is not None:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
        if rlist:
            key = os.read(sys.stdin.fileno(), 1).decode('utf-8', errors='ignore')
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
        return key
    else:
        rlist, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
        if rlist:
            return os.read(sys.stdin.fileno(), 1).decode('utf-8', errors='ignore')
        return ''


def main():
    settings = None
    if sys.stdin.isatty():
        try:
            settings = termios.tcgetattr(sys.stdin)
        except Exception:
            settings = None

    rclpy.init()
    node = rclpy.create_node('teleop_keyboard_node')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)

    linear_speed = 0.07   # m/s (Tốc độ chuẩn 7 cm/s cho động cơ 775 có hộp số)
    angular_speed = 0.25  # rad/s (Góc xoay êm, không vọt bánh)

    target_x = 0.0
    target_th = 0.0
    last_action = "DỪNG [Space]"

    print(BANNER)
    print(f"👉 CHẾ ĐỘ CHẠY LIÊN TỤC: Bấm [W] hoặc [i] là xe chạy liên tục cho đến khi bấm [Space]/[X]/[K] để dừng.")
    print(f"👉 Mức tốc độ: Dài = {linear_speed:.2f} m/s | Góc = {angular_speed:.2f} rad/s\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = get_key(settings, timeout=0.05)

            if key in MOVE_BINDINGS:
                target_x = MOVE_BINDINGS[key][0]
                target_th = MOVE_BINDINGS[key][1]

                if target_x > 0 and target_th == 0: last_action = "TIẾN ⬆️ [W/i]"
                elif target_x < 0 and target_th == 0: last_action = "LÙI ⬇️ [S/,]"
                elif target_th > 0 and target_x == 0: last_action = "XOAY TRÁI ⬅️ [A/j]"
                elif target_th < 0 and target_x == 0: last_action = "XOAY PHẢI ➡️ [D/l]"
                elif target_x > 0 and target_th > 0: last_action = "TIẾN TRÁI ↖️ [Q/u]"
                elif target_x > 0 and target_th < 0: last_action = "TIẾN PHẢI ↗️ [E/o]"
                elif target_x < 0 and target_th < 0: last_action = "LÙI TRÁI ↙️ [Z/m]"
                elif target_x < 0 and target_th > 0: last_action = "LÙI PHẢI ↘️ [C/.]"
                else: last_action = "DỪNG [Space]"

            elif key in SPEED_BINDINGS:
                factor = SPEED_BINDINGS[key]
                linear_speed = round(max(0.03, min(0.50, linear_speed * factor)), 3)
                angular_speed = round(max(0.08, min(1.20, angular_speed * factor)), 2)
                print(f"\n⚡ [TỐC ĐỘ MỚI] Dài: {linear_speed:.3f} m/s | Góc: {angular_speed:.2f} rad/s")

            elif key == '\x03':  # Ctrl+C
                break

            # Phát liên tục ~20 Hz duy trì vận tốc đều đặn, nuôi ESP32 Watchdog
            twist = Twist()
            twist.linear.x = target_x * linear_speed
            twist.angular.z = target_th * angular_speed
            pub.publish(twist)

            print(f"\r[ĐIỀU KHIỂN] {last_action:<20} | v = {twist.linear.x:+.2f} m/s (Mức: {linear_speed:.2f} m/s) | w = {twist.angular.z:+.2f} rad/s", end="", flush=True)

    except Exception as e:
        print(f"\nLỗi: {e}")

    finally:
        twist = Twist()
        pub.publish(twist)
        if settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()
        print("\n\nĐã dừng xe và thoát teleop an toàn.")


if __name__ == '__main__':
    main()
