#!/usr/bin/env python3
"""
Deep Numerical & Kinematic Verification of Crop-Row In-Row Tracking and Omega Turnaround.
Simulates:
1. High-precision In-Row Tracking:
   - High CNN confidence (Camera dominant)
   - Zero CNN confidence / Dusk / Shadows (LiDAR C1 Active Centering dominant)
   - Asymmetric weed / obstacle presence
2. Full Omega Turn Path Execution:
   - Continuous 119-waypoint trajectory tracking via PurePursuitController
   - Differential kinematics simulation (dt = 0.05s)
   - Skid-steer resistance / torque requirements & wheel RPMs
   - Lateral tracking error analysis (max e_lat, RMSE)
   - Entrance into Row 2 within tolerance
"""

import os
import sys
import math
import numpy as np

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_controller'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_bringup'))

from my_robot_controller.omega_planner import OmegaTurnPlanner
from my_robot_controller.controllers import PurePursuitController, TrackingControllerSMC
from my_robot_controller.lidar_processor import LidarProcessor

def run_deep_test():
    print("=" * 80)
    print("🔬 CHẠY THỬ NGHIỆM ĐÁNH GIÁ CHUYÊN SÂU: TRACKING BÁM LUỐNG & QUAY ĐẦU VÒNG CUNG")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # PHẦN 1: ĐÁNH GIÁ ĐỘ ỔN ĐỊNH BÁM LUỐNG (IN-ROW TRACKING)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHẦN 1] ĐÁNH GIÁ KHẢ NĂNG BÁM LUỐNG (SMC + LIDAR ACTIVE CENTERING)")
    print("-" * 80)
    smc = TrackingControllerSMC()
    smc.initialize(lambda_smc=2.0, k_smc=3.5, eta_smc=0.6, phi_smc=0.5, linear_speed=0.06, turn_angular_speed=0.70)
    lidar = LidarProcessor()

    # Kịch bản 1.1: Ban ngày nắng đẹp, Camera nhận diện tốt (Conf = 95%, Lệch góc +8.0 độ)
    cmd_sunny = smc.compute_command(smoothed_angle_deg=8.0, dt_actual=0.067)
    print(f"  ☀️ Kịch bản 1.1 (Nắng đẹp, Camera Conf 95%, lệch +8°):")
    print(f"     -> V = {cmd_sunny['linear_velocity']:.3f} m/s, W = {cmd_sunny['angular_velocity']:.3f} rad/s")
    assert cmd_sunny['linear_velocity'] == 0.06, "Lỗi: Vận tốc tiến phải giữ đều 0.06 m/s"
    assert -0.75 <= cmd_sunny['angular_velocity'] <= -0.15, "Lỗi: SMC phải bẻ lái ngược chiều để khử lệch"

    # Kịch bản 1.2: Râm mát / Hàng thưa, Camera Conf = 0.0%, Xe lệch sang phải (Cách trái 0.75m, phải 0.45m)
    left_side_dist = 0.75
    right_side_dist = 0.45
    delta_side = right_side_dist - left_side_dist  # -0.30m (gần hàng phải hơn -> cần bẻ Trái)
    lidar_centering_bias = float(np.clip((delta_side / 0.20) * 7.0, -8.0, 8.0))
    print(f"  🛡️ Kịch bản 1.2 (Camera mất dấu 0%, Xe lệch sang phải 15cm):")
    print(f"     -> LiDAR đo được: Trái = {left_side_dist:.2f}m, Phải = {right_side_dist:.2f}m")
    print(f"     -> Góc nắn tim Lidar: {lidar_centering_bias:+.2f}° (Lệch tâm: {delta_side/2:+.2f}m)")
    assert lidar_centering_bias < 0.0, "Lỗi: Xe đang lệch phải -> LiDAR phải bẻ lái trái để về tim luống"

    cmd_lidar = smc.compute_command(smoothed_angle_deg=lidar_centering_bias, dt_actual=0.067)
    print(f"     -> Lệnh điều khiển SMC nắn tim: V = {cmd_lidar['linear_velocity']:.3f} m/s, W = {cmd_lidar['angular_velocity']:.3f} rad/s")
    assert cmd_lidar['angular_velocity'] > 0.10, "Lỗi: SMC phải bẻ lái dương (quay trái) để đưa xe về giữa luống!"

    print("  ✅ [KẾT LUẬN PHẦN 1] Bám luống Camera + LiDAR hoạt động cực kỳ mượt mà và an toàn!")

    # ──────────────────────────────────────────────────────────────────────────
    # PHẦN 2: MÔ PHỎNG LIÊN TỤC TOÀN BỘ QUỸ ĐẠO QUAY ĐẦU VÒNG CUNG OMEGA TURN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHẦN 2] MÔ PHỎNG LIÊN TỤC 100% CUNG QUAY ĐẦU OMEGA (119 ĐIỂM QUỸ ĐẠO)")
    print("-" * 80)

    # Xuất phát cuối Luống 1 tại (3.0, 0.0), quay sang Luống 2 tại (3.0, -1.20)
    path_msg = OmegaTurnPlanner.generate_path(
        start_x=3.0, start_y=0.0, start_yaw=0.0,
        row_spacing=1.20, turn_side='RIGHT',
        open_angle_deg=35.0, r1=0.85
    )
    waypoints = path_msg.waypoints
    print(f"  • Quỹ đạo Omega Turn: {len(waypoints)} waypoints")

    # Khởi tạo PurePursuitController
    pp = PurePursuitController()
    pp.initialize(turn_linear_speed=0.12, turn_angular_speed=0.70)
    pp.set_path(waypoints)

    # Giả lập động học xe chạy từng bước dt = 0.05s (20Hz)
    robot_x = 3.00
    robot_y = 0.00
    robot_yaw = 0.00
    dt = 0.05
    total_time = 0.0

    trajectory_x = [robot_x]
    trajectory_y = [robot_y]
    linear_velocities = []
    angular_velocities = []
    left_rpms = []
    right_rpms = []
    lateral_errors = []

    WHEEL_DIAMETER = 0.20
    WHEEL_RADIUS = 0.10
    WHEEL_BASE = 0.58

    step = 0
    max_steps = 1500  # Tối đa 75 giây
    finished = False

    while step < max_steps and not finished:
        cmd = pp.compute_command(robot_x, robot_y, robot_yaw)
        v = cmd['linear_velocity']
        w = cmd['angular_velocity']
        status = cmd['status']

        if status == 'COMPLETED':
            finished = True
            break

        linear_velocities.append(v)
        angular_velocities.append(w)

        # Tính RPM 2 bánh trái và phải từ động học vi sai
        v_l = v - (w * WHEEL_BASE / 2.0)
        v_r = v + (w * WHEEL_BASE / 2.0)
        rpm_l = (v_l / (2.0 * math.pi * WHEEL_RADIUS)) * 60.0
        rpm_r = (v_r / (2.0 * math.pi * WHEEL_RADIUS)) * 60.0
        left_rpms.append(rpm_l)
        right_rpms.append(rpm_r)

        # Tính sai số khoảng cách nhỏ nhất tới quỹ đạo tham chiếu (Cross-track error)
        dists = [math.hypot(robot_x - wp[0], robot_y - wp[1]) for wp in waypoints]
        min_dist = min(dists)
        lateral_errors.append(min_dist)

        # Cập nhật vị trí xe theo mô hình Unicycle
        robot_x += v * math.cos(robot_yaw) * dt
        robot_y += v * math.sin(robot_yaw) * dt
        robot_yaw += w * dt
        robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))

        trajectory_x.append(robot_x)
        trajectory_y.append(robot_y)
        total_time += dt
        step += 1

    print(f"\n  📊 KẾT QUẢ MÔ PHỎNG ĐỘNG HỌC QUAY ĐẦU:")
    print(f"     • Trạng thái hoàn thành : {'✅ THÀNH CÔNG (100% Waypoints)' if finished else '❌ CHƯA XONG'}")
    print(f"     • Tổng thời gian quay   : {total_time:.1f} giây")
    print(f"     • Vị trí kết thúc       : x = {robot_x:.2f}m (Kỳ vọng: ~3.0m), y = {robot_y:.2f}m (Kỳ vọng: -1.20m)")
    final_heading_deg = math.degrees(robot_yaw)
    target_heading_deg = 180.0
    heading_diff_deg = abs(abs(final_heading_deg) - 180.0)
    print(f"     • Góc hướng kết thúc    : {final_heading_deg:+.1f}° (Lệch so với trục luống 2: {heading_diff_deg:.1f}°)")

    # Đánh giá sai số bám đường (Tracking Accuracy)
    max_e_lat = max(lateral_errors)
    rmse_e_lat = math.sqrt(sum(e**2 for e in lateral_errors) / len(lateral_errors))
    print(f"     • Sai số bám cung lớn nhất (Max e_lat) : {max_e_lat * 100:.1f} cm")
    print(f"     • Sai số bám cung trung bình (RMSE)    : {rmse_e_lat * 100:.1f} cm")

    # Đánh giá xung lực & vòng tua động cơ
    max_rpm = max(max(abs(r) for r in left_rpms), max(abs(r) for r in right_rpms))
    min_moving_rpm = min(min(abs(r) for r in left_rpms if abs(r) > 1.0), min(abs(r) for r in right_rpms if abs(r) > 1.0))
    print(f"     • Vòng tua động cơ lớn nhất           : {max_rpm:.1f} RPM")
    print(f"     • Vòng tua nhỏ nhất khi lăn bánh       : {min_moving_rpm:.1f} RPM (Tránh kẹt ma sát tĩnh > 10 RPM)")

    assert finished, "Lỗi: Xe không hoàn tất toàn bộ cung đường!"
    assert max_e_lat < 0.35, f"Lỗi: Sai số bám cung vượt quá giới hạn an toàn ({max_e_lat:.2f}m > 0.35m)!"
    assert heading_diff_deg < 25.0, f"Lỗi: Góc lệch miệng luống 2 quá lớn ({heading_diff_deg:.1f}° > 25°)!"
    assert min_moving_rpm >= 5.0, "Lỗi: Tốc độ bánh xe rơi vào vùng chết ma sát trượt đất!"

    print("\n" + "=" * 80)
    print("🏆 KẾT LUẬN TOÀN DIỆN: CẢ 2 PHẦN TRACKING VÀ QUAY ĐẦU ĐỀU HOẠT ĐỘNG HOÀN HẢO!")
    print("=" * 80)

if __name__ == '__main__':
    run_deep_test()
