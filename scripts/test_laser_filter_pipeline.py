#!/usr/bin/env python3
"""
Test script to verify the LaserFilter pipeline.
Publishes a simulated scan on /scan_raw containing:
- Pillar points at ~0.10m directly behind the LiDAR (angle ~0 rad in laser_frame)
- Chassis/screw points at < 0.18m
- Genuine obstacles in front (angle ~pi) and sides (angle ~pi/2, -pi/2) at 1.5m - 2.5m

Subscribes to /scan to verify that:
- Pillar and chassis points are filtered to inf
- Genuine obstacles are preserved
"""

import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan


class LaserFilterTester(Node):
    def __init__(self):
        super().__init__('laser_filter_tester')

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub = self.create_publisher(LaserScan, '/scan_raw', qos_sensor)
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_sensor)

        self.received = False
        self.test_success = False

        # Timer to publish mock scan at 10Hz
        self.timer = self.create_timer(0.1, self.publish_mock_scan)

    def publish_mock_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = 2.0 * math.pi / 360.0
        msg.time_increment = 0.0001
        msg.scan_time = 0.1
        msg.range_min = 0.05
        msg.range_max = 12.0

        n = 360
        ranges = [3.0] * n  # default 3m open space

        # 1. Hardware screw/chassis reflection at angle -90 deg (270 deg) at 0.12m
        ranges[90] = 0.12

        # 2. Pillar directly behind LiDAR (around angle 0 rad) at 0.09m
        # Indices around 180 (since angles go from -pi to pi, index 180 is angle 0)
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            ranges[180 + offset] = 0.09

        # 3. Genuine obstacle in front (angle pi / -pi, indices 0 and 359) at 1.8m
        ranges[0] = 1.80
        ranges[359] = 1.80

        # 4. Left wall/tree row at angle 90 deg (index 270) at 0.65m
        ranges[270] = 0.65

        msg.ranges = ranges
        msg.intensities = [47.0] * n
        self.pub.publish(msg)

    def scan_cb(self, msg: LaserScan):
        if self.received:
            return
        self.received = True

        r = msg.ranges
        print("\n" + "=" * 60)
        print("📊 [KẾT QUẢ KIỂM THỬ BỘ LỌC TIA LASER (/scan)]")
        print("=" * 60)

        # Kiểm tra điểm cột chống (index 180 - góc sau xe)
        pillar_val = r[180]
        pillar_ok = math.isinf(pillar_val)
        print(f" • Cột chống màu trắng (cự ly 0.09m sau LiDAR) : {'ĐÃ LỌC THÀNH CÔNG (inf)' if pillar_ok else f'THẤT BẠI ({pillar_val})'}")

        # Kiểm tra điểm chân đế/vỏ hộp (index 90)
        chassis_val = r[90]
        chassis_ok = math.isinf(chassis_val)
        print(f" • Chân đế/ốc vít sát máy (cự ly 0.12m)        : {'ĐÃ LỌC THÀNH CÔNG (inf)' if chassis_ok else f'THẤT BẠI ({chassis_val})'}")

        # Kiểm tra vật cản thật phía trước (index 0)
        front_val = r[0]
        front_ok = abs(front_val - 1.80) < 0.01
        print(f" • Vật cản thật phía trước (cự ly 1.80m)       : {'BẢO TOÀN NGUYÊN VẸN' if front_ok else f'SAI ({front_val})'}")

        # Kiểm tra mép luống cây bên sườn (index 270)
        side_val = r[270]
        side_ok = abs(side_val - 0.65) < 0.01
        print(f" • Hàng cây bên sườn xe (cự ly 0.65m)          : {'BẢO TOÀN NGUYÊN VẸN' if side_ok else f'SAI ({side_val})'}")

        print("=" * 60)
        if pillar_ok and chassis_ok and front_ok and side_ok:
            print("🎉 TẤT CẢ ĐIỀU KIỆN ĐẠT CHUẨN 100%! CỘT TRẮNG KHÔNG CÒN GÂY NHIỄU.")
            self.test_success = True
        else:
            print("❌ CÓ ĐIỀU KIỆN CHƯA ĐẠT.")
            self.test_success = False


def main():
    rclpy.init()
    tester = LaserFilterTester()

    start_time = tester.get_clock().now()
    while rclpy.ok() and not tester.received:
        rclpy.spin_once(tester, timeout_sec=0.1)
        elapsed = (tester.get_clock().now() - start_time).nanoseconds / 1e9
        if elapsed > 4.0:
            print("⏱️ Hết thời gian chờ dữ liệu từ /scan (timeout)!")
            break

    tester.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if tester.test_success else 1)


if __name__ == '__main__':
    main()
