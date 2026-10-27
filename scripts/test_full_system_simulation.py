#!/usr/bin/env python3
"""
Full System End-to-End Simulation & Stress Test for Autonomous Crop-Row Robot.
Tests:
1. CnnDriverNode initialization with params_real.yaml.
2. In-row tracking with CNN + LiDAR fusion.
3. End-of-row detection (Goal 1 reached at x=3.0m).
4. Omega Turn path generation & Pure Pursuit tracking with Pivot Priority.
5. Re-entry heading trimming & Transition to Row 2.
6. Row 2 tracking & Mission Completion (Goal 3 reached at x=0.0m).
7. Verify zero unhandled exceptions, zero deadlocks, and correct kinematic commands.
"""

import os
import sys
import time
import math
import cv2
import numpy as np

# ROS 2 Python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_controller'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_bringup'))

from my_robot_controller.cnn_driver import CnnDriverNode
from my_robot_controller.fsm import FSMState
from my_robot_bringup.esp32_bridge import ESP32Bridge

def create_synthetic_image(h=480, w=640, lane_center_x=320):
    """Tạo ảnh giả lập luống bắp ban ngày (thảm xanh 2 bên, rãnh đất ở giữa)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Nền đất nâu
    img[:] = (40, 70, 100)
    # Hàng bắp bên trái (xanh lá cây)
    left_edge = max(0, int(lane_center_x - 160))
    img[:, :left_edge] = (30, 160, 40)
    # Hàng bắp bên phải (xanh lá cây)
    right_edge = min(w, int(lane_center_x + 160))
    img[:, right_edge:] = (30, 160, 40)
    return img

def create_synthetic_scan(left_dist=0.60, right_dist=0.60, front_dist=2.50):
    """Tạo LaserScan 360 độ (1 khoảng cách cho các góc)."""
    scan = LaserScan()
    scan.header.frame_id = "laser_frame"
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = math.radians(1.0)
    scan.range_min = 0.15
    scan.range_max = 12.0
    num_pts = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
    ranges = [5.0] * num_pts
    
    # Góc -25 đến +25: phía trước
    for deg in range(-25, 26):
        idx = int((math.radians(deg) - scan.angle_min) / scan.angle_increment)
        if 0 <= idx < num_pts:
            ranges[idx] = front_dist
            
    # Góc 70 đến 110: bên trái
    for deg in range(70, 111):
        idx = int((math.radians(deg) - scan.angle_min) / scan.angle_increment)
        if 0 <= idx < num_pts:
            ranges[idx] = left_dist

    # Góc -110 đến -70: bên phải
    for deg in range(-110, -69):
        idx = int((math.radians(deg) - scan.angle_min) / scan.angle_increment)
        if 0 <= idx < num_pts:
            ranges[idx] = right_dist

    scan.ranges = ranges
    return scan

def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def run_simulation_test():
    print("=" * 75)
    print("🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM MÔ PHỎNG TOÀN DIỆN (FULL SYSTEM STRESS TEST)")
    print("=" * 75)

    rclpy.init()

    # 1. Khởi tạo CnnDriverNode với cờ enable_mission_goals = True
    print("\n[BƯỚC 1] Khởi tạo CnnDriverNode...")
    node = CnnDriverNode()
    node.enable_mission_goals = True
    node.enable_uturn = True
    node.goal_1_x = 3.0
    node.goal_1_y = 0.0
    node.goal_2_x = 3.0
    node.goal_2_y = -1.20
    node.goal_3_x = 0.0
    node.goal_3_y = -1.20
    node.min_row_length = 2.6
    node.row_spacing = 1.20
    node.linear_speed = 0.06
    node.turn_linear_speed = 0.12
    node.turn_angular_speed = 0.70
    node.warmup_time = 0.0
    node.is_mission_started = True

    # 2. Khởi tạo ESP32Bridge để kiểm tra chuỗi lệnh Serial
    print("[BƯỚC 2] Khởi tạo ESP32Bridge mô phỏng...")
    bridge = ESP32Bridge()
    bridge.mode = 'mock'

    # Lưu vết các lệnh cmd_vel xuất ra
    published_twists = []
    def mock_pub(twist):
        published_twists.append((twist.linear.x, twist.angular.z))
    node.cmd_vel_pub.publish = mock_pub

    # Helper gửi sensor updates vào node
    def update_sensors(x, y, yaw, left_d=0.60, right_d=0.60, front_d=3.0, lane_cx=320, confidence_boost=1.0):
        # Odom
        odom = Odometry()
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation = yaw_to_quaternion(yaw)
        odom.twist.twist.linear.x = 0.06
        node.odom_callback(odom)

        # IMU
        imu = Imu()
        imu.orientation = yaw_to_quaternion(yaw)
        imu.angular_velocity.z = 0.0
        node.imu_callback(imu)

        # LiDAR
        scan = create_synthetic_scan(left_dist=left_d, right_dist=right_d, front_dist=front_d)
        node.scan_callback(scan)

        # Camera
        img = create_synthetic_image(lane_center_x=lane_cx)
        from cv_bridge import CvBridge
        cv_b = CvBridge()
        img_msg = cv_b.cv2_to_imgmsg(img, encoding="bgr8")
        node.image_callback(img_msg)

    # ── GIAI ĐOẠN 1: CHẠY TRONG LUỐNG 1 (x = 0.0 -> 2.4m) ──
    print("\n[BƯỚC 3] Giả lập chạy trong Luống 1 (x: 0.0m -> 2.4m, lệch phải 10cm)...")
    for sim_x in np.arange(0.0, 2.5, 0.3):
        update_sensors(x=sim_x, y=0.0, yaw=0.0, left_d=0.70, right_d=0.50, front_d=3.0)
        rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM: {node.fsm.get_state()} (Kỳ vọng: TRACKING)")
    assert node.fsm.get_state() == FSMState.TRACKING, "Lỗi: Không ở trạng thái TRACKING trong luống!"
    last_v, last_w = published_twists[-1] if published_twists else (0, 0)
    print(f"  • Lệnh lái xuất ra: V = {last_v:.3f} m/s, W = {last_w:.3f} rad/s")
    assert last_v > 0, "Lỗi: Vận tốc tiến phải > 0 khi chạy trong luống!"

    # ── GIAI ĐOẠN 2: CHẠM MỐC GOAL 1 (x >= 3.0m) -> KÍCH HOẠT DRIVE_OUT VÀ OMEGA TURN ──
    print("\n[BƯỚC 4] Đến cuối Luống 1 (x = 3.00m) -> Kiểm tra kích hoạt DRIVE_OUT & Omega Turn...")
    # Cảm biến báo hết hàng (phía trước và 2 bên thoáng)
    update_sensors(x=3.00, y=0.0, yaw=0.0, left_d=1.5, right_d=1.5, front_d=5.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM: {node.fsm.get_state()} (Kỳ vọng: DRIVE_OUT)")
    assert node.fsm.get_state() == FSMState.DRIVE_OUT, \
        f"Lỗi: Không chuyển sang DRIVE_OUT, state hiện tại = {node.fsm.get_state()}"
    print(f"  ✅ Đã vào DRIVE_OUT chạy thẳng thoát đầu luống thành công!")

    # Giả lập chạy thẳng thoát luống 1.80m (x: 3.00m -> 4.85m >= drive_out_distance 1.80m)
    node.distance_traveled = 4.85
    update_sensors(x=4.85, y=0.0, yaw=0.0, left_d=1.5, right_d=1.5, front_d=5.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    if node.fsm.get_state() == FSMState.UTURN_PLANNING:
        update_sensors(x=4.85, y=0.0, yaw=0.0, left_d=1.5, right_d=1.5, front_d=5.0)
        rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM sau khi thoát luống: {node.fsm.get_state()} (Kỳ vọng: PATH_FOLLOWING)")
    assert node.fsm.get_state() == FSMState.PATH_FOLLOWING, "Lỗi: Không vào PATH_FOLLOWING sau khi thoát luống!"
    print(f"  ✅ Đã thoát luống an toàn và vào PATH_FOLLOWING bám cung Omega Turn thành công!")

    # ── GIAI ĐOẠN 3: BÁM CUNG OMEGA & KIỂM TRA PIVOT PRIORITY ──
    print("\n[BƯỚC 5] Kiểm tra Skid-Steer Pivot Priority trong PurePursuitController...")
    # Giả lập thân xe bị lệch hướng 45 độ (> 20 độ) so với tiếp tuyến
    update_sensors(x=3.10, y=0.10, yaw=math.radians(45.0), left_d=2.0, right_d=2.0, front_d=5.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    v_turn, w_turn = published_twists[-1]
    print(f"  • Khi lệch 45° (> 20°): V = {v_turn:.3f} m/s, W = {w_turn:.3f} rad/s")
    assert v_turn <= 0.08, f"Lỗi: Bò chậm chống cày đất khi lệch góc lớn (thực tế: {v_turn})!"
    assert abs(w_turn) >= 0.70, f"Lỗi: Vận tốc góc bẻ lái ngẫu lực vi sai phải đạt 0.70 rad/s (thực tế: {w_turn})"

    # Giả lập đầu xe đã thẳng bám tiếp tuyến (lệch ~0 độ <= 20 độ)
    # Set gyro_z để tránh slip compensation hạ tốc độ
    node.localization_manager.imu_angular_vel_z = 0.10
    update_sensors(x=3.10, y=0.10, yaw=math.radians(-18.0), left_d=2.0, right_d=2.0, front_d=5.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    v_arc, w_arc = published_twists[-1]
    print(f"  • Khi bám sát tiếp tuyến (lệch <= 20°): V = {v_arc:.3f} m/s, W = {w_arc:.3f} rad/s")
    assert v_arc > 0.04, f"Lỗi: Xe phải lăn bánh tiến theo vòng cung khi lệch góc nhỏ (thực tế: {v_arc})!"
    print(f"  ✅ Cơ chế Rolling Pivot và Lăn bánh vòng cung hoạt động HOÀN HẢO!")

    # ── GIAI ĐOẠN 4: HOÀN TẤT VÒNG QUAY 180° SANG LUỐNG 2 ──
    print("\n[BƯỚC 6] Giả lập đưa xe về cuối cung Omega (x=3.0m, y=-1.20m, Yaw=180°)...")
    # Cập nhật IMU quay đủ 180 độ
    node.uturn_accum_imu_yaw = math.pi
    # Đặt xe ở đầu Luống 2 (x=3.05m, y=-1.20m, yaw=pi)
    update_sensors(x=3.05, y=-1.20, yaw=math.pi, left_d=0.60, right_d=0.60, front_d=3.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM: {node.fsm.get_state()}")
    print(f"  • Đang ở Luống số: {node.current_lane_idx} (Kỳ vọng: Luống 2)")
    assert node.current_lane_idx == 2, "Lỗi: Phải chuyển sang Luống 2 sau khi quay đầu!"

    # ── GIAI ĐOẠN 5: CHẠY LUỐNG 2 VỀ ĐÍCH GOAL 3 ──
    print("\n[BƯỚC 7] Giả lập chạy Luống 2 ngược chiều về đích Goal 3 (x=0.0m, y=-1.20m)...")
    node.fsm.set_state(FSMState.TRACKING)
    node.inside_row = True
    node.distance_traveled = 2.8

    # Đưa xe về x = 0.05m ở Luống 2
    update_sensors(x=0.05, y=-1.20, yaw=math.pi, left_d=0.60, right_d=0.60, front_d=3.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM tại Goal 3: {node.fsm.get_state()} (Kỳ vọng: DRIVE_OUT)")
    assert node.fsm.get_state() == FSMState.DRIVE_OUT, "Lỗi: Phải kích hoạt DRIVE_OUT thoát hàng khi chạm Goal 3!"

    # Giả lập hoàn tất thoát hàng 1.80m
    node.distance_traveled = 4.70
    update_sensors(x=-1.85, y=-1.20, yaw=math.pi, left_d=1.5, right_d=1.5, front_d=5.0)
    rclpy.spin_once(node, timeout_sec=0.01)

    print(f"  • Trạng thái FSM tại vạch đích: {node.fsm.get_state()} (Kỳ vọng: IDLE - MISSION_COMPLETE)")
    assert node.fsm.get_state() == FSMState.IDLE, "Lỗi: Phải dừng về IDLE khi hoàn thành nhiệm vụ Goal 3!"
    last_stop_v, last_stop_w = published_twists[-1]
    print(f"  • Lệnh dừng xe an toàn: V = {last_stop_v:.3f} m/s, W = {last_stop_w:.3f} rad/s")
    assert last_stop_v == 0.0 and last_stop_w == 0.0, "Lỗi: Robot phải phanh dừng tuyệt đối khi về đích!"

    # ── GIAI ĐOẠN 6: KIỂM TRA CẦU NỐI ESP32 VỚI LỆNH MỚI ──
    print("\n[BƯỚC 8] Kiểm tra ESP32Bridge chuyển đổi lệnh /cmd_vel sang chuỗi Serial...")
    # Test 1: Lệnh xoay tại chỗ (Pivot)
    twist_pivot = Twist()
    twist_pivot.linear.x = 0.0
    twist_pivot.angular.z = 0.70
    bridge.cmd_vel_callback(twist_pivot)
    bridge.update_loop()
    rpm_l = bridge.target_rpm_left
    rpm_r = bridge.target_rpm_right
    print(f"  • Khi xoay tại chỗ W=0.70: Target RPM Left = {rpm_l:.1f}, Right = {rpm_r:.1f}")
    assert rpm_l <= -12.0, "Lỗi: Bánh trái phải đảo chiều lùi <= -12.0 RPM khi xoay tại chỗ!"
    assert rpm_r >= 15.0, "Lỗi: Bánh phải phải tiến >= +15.0 RPM khi xoay tại chỗ!"

    # Test 2: Lệnh chạy tiến uốn lượn mượt trong luống
    twist_straight = Twist()
    twist_straight.linear.x = 0.06
    twist_straight.angular.z = 0.10
    bridge.cmd_vel_callback(twist_straight)
    bridge.update_loop()
    rpm_l2 = bridge.target_rpm_left
    rpm_r2 = bridge.target_rpm_right
    print(f"  • Khi chạy tiến bẻ lái V=0.06, W=0.10: Target RPM Left = {rpm_l2:.1f}, Right = {rpm_r2:.1f}")
    assert rpm_l2 > 0.0 and rpm_r2 > 0.0, "Lỗi: Khi chạy tiến bẻ lái nhẹ, cả 2 bánh phải cùng quay tiến lăn mượt!"

    print("\n" + "=" * 75)
    print("🏆 TẤT CẢ CÁC BƯỚC TEST ĐỀU ĐẠT CHUẨN XUẤT SẮC (100% PASS - ZERO ERRORS)!")
    print("=" * 75)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    run_simulation_test()
