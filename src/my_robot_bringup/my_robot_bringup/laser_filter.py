#!/usr/bin/env python3
"""
ROS 2 LaserScan Filter Node for RPLIDAR C1 / C1M1.
Filters out:
1. Robot hardware blind zone (chassis / mounting base / screws within radius < min_distance).
2. Vertical camera support pillar & cable shadow located directly behind the LiDAR.
3. Out-of-bounds readings (> max_distance).

Publishes clean LaserScan data to /scan for SLAM, Costmap, Nav2, and AI navigation.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan


class LaserFilterNode(Node):
    def __init__(self):
        super().__init__('laser_filter_node')

        # Parameters
        self.declare_parameter('input_topic', '/scan_raw')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('min_distance', 0.18)       # m — Lọc bán kính chân đế & cột đỡ (< 18cm)
        self.declare_parameter('max_distance', 12.0)       # m — Tầm xa tối đa RPLIDAR C1
        self.declare_parameter('filter_rear_cone', True)   # Lọc góc sau xe (hướng cột camera RealSense)
        self.declare_parameter('rear_angle_deg', 30.0)     # ±30 độ phía đuôi xe (nón 60 độ sau lưng)
        self.declare_parameter('rear_dist_limit', 0.35)    # m — Chỉ lọc điểm trong phạm vi 35cm phía sau xe
        self.declare_parameter('log_stats', False)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.min_dist = float(self.get_parameter('min_distance').value)
        self.max_dist = float(self.get_parameter('max_distance').value)
        self.filter_rear = bool(self.get_parameter('filter_rear_cone').value)
        self.rear_half_rad = math.radians(float(self.get_parameter('rear_angle_deg').value))
        self.rear_limit = float(self.get_parameter('rear_dist_limit').value)
        self.log_stats = bool(self.get_parameter('log_stats').value)

        # QoS sensor data
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self.scan_callback,
            qos_sensor
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            self.output_topic,
            qos_sensor
        )

        self.filtered_count = 0
        self.total_frames = 0

        self.get_logger().info(
            f"🛡️ [LaserFilter] Khởi động thành công! "
            f"({self.input_topic} ➜ {self.output_topic} | min_dist={self.min_dist:.2f}m | "
            f"rear_cone=±{math.degrees(self.rear_half_rad):.1f}° < {self.rear_limit:.2f}m)"
        )

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return

        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment

        # Normalize angles to [-pi, pi]
        # In laser_frame with rpy="0 0 pi", angle 0 points to robot REAR (-X in base_link)
        norm_angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

        # Mask 1: Điểm quá gần chân đế / ốc vít / cột chống (< min_distance)
        blind_mask = np.isfinite(ranges) & (ranges < self.min_dist)

        # Mask 2: Điểm quá xa (> max_distance)
        far_mask = np.isfinite(ranges) & (ranges > self.max_dist)

        # Mask 3: Góc nón phía sau thân xe (vị trí cây cột trắng & dây cáp)
        if self.filter_rear:
            rear_mask = (
                np.isfinite(ranges)
                & (np.abs(norm_angles) <= self.rear_half_rad)
                & (ranges <= self.rear_limit)
            )
        else:
            rear_mask = np.zeros(n, dtype=bool)

        combined_filter = blind_mask | far_mask | rear_mask

        # Gán inf cho các điểm bị che/nhiễu
        ranges[combined_filter] = np.inf

        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = max(msg.range_min, self.min_dist)
        filtered_msg.range_max = min(msg.range_max, self.max_dist)
        filtered_msg.ranges = ranges.tolist()

        if len(msg.intensities) == n:
            intensities = np.array(msg.intensities, dtype=np.float32)
            intensities[combined_filter] = 0.0
            filtered_msg.intensities = intensities.tolist()
        else:
            filtered_msg.intensities = msg.intensities

        self.scan_pub.publish(filtered_msg)

        if self.log_stats:
            pts_filtered = int(np.sum(combined_filter))
            self.filtered_count += pts_filtered
            self.total_frames += 1
            if self.total_frames % 50 == 0:
                self.get_logger().info(f"[LaserFilter] Đã lọc {pts_filtered}/{n} tia quét từ cột chống/chân đế.")


def main(args=None):
    rclpy.init(args=args)
    node = LaserFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
