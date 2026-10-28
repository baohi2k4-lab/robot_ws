#!/usr/bin/env python3
"""
CNN Driver Node — ROS 2 controller cho robot tự hành giữa hàng bắp.

Tách biệt các khối theo mô hình: Perception ↓ Decision ↓ Planning ↓ Control ↓ Hardware.
- Decision: FSMCoordinator (fsm.py)
- Planning: OmegaTurnPlanner (omega_planner.py) & Quintic Row Avoidance
- Control: TrackingControllerSMC, PurePursuitController (controllers.py)
"""

import os
import sys
import time
import math
import cv2
import rclpy
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, LaserScan, NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, Empty
import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from my_robot_controller.inference_handler import InferenceHandler
from my_robot_controller.interfaces import ControllerInterface
from my_robot_controller.controllers import TrackingControllerSMC, PurePursuitController, ControllerManager, SafetyController
from my_robot_controller.fsm import FSMCoordinator, FSMState, FSMEvent
from my_robot_controller.path_utils import Path
from my_robot_controller.omega_planner import OmegaTurnPlanner
from my_robot_controller.lidar_processor import LidarProcessor
from my_robot_controller.perception_manager import PerceptionManager, EndOfRowDetector
from my_robot_controller.localization_manager import LocalizationManager
from geometry_msgs.msg import TwistStamped
from my_robot_controller.telemetry_logger import TelemetryLogger
from my_robot_controller.omega_evaluator import OmegaTurnEvaluator
from my_robot_controller.yolo_arrow_detector import YoloArrowDetector



class CnnDriverNode(Node):
    def __init__(self):
        super().__init__('cnn_driver_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('model_path', 'models/crop_row_cnn_best.onnx')
        self.declare_parameter('input_height', 384)
        self.declare_parameter('input_width', 384)
        self.declare_parameter('num_threads', 0)
        self.declare_parameter('mask_threshold', 0.35)
        self.declare_parameter('linear_speed', 0.07)
        self.declare_parameter('turn_linear_speed', 0.07)
        self.declare_parameter('turn_angular_speed', 0.42)
        self.declare_parameter('yellow_arrow_turn_angle_deg', 90.0)
        self.declare_parameter('low_confidence_threshold', 0.35)
        self.declare_parameter('high_confidence_threshold', 0.50)
        self.declare_parameter('lambda_smc', 2.0)
        self.declare_parameter('k_smc', 3.5)
        self.declare_parameter('eta_smc', 0.6)
        self.declare_parameter('phi_smc', 0.5)
        self.declare_parameter('max_steering_angle_deg', 16.0)
        self.declare_parameter('turn_in_place_threshold_deg', 30.0)
        self.declare_parameter('turn_in_place_resume_deg', 10.0)
        self.declare_parameter('row_spacing', 0.90)
        self.declare_parameter('ema_alpha', 0.45)
        self.declare_parameter('enable_uturn', False)
        self.declare_parameter('warmup_time', 1.0)
        self.declare_parameter('navigation_mode', 'auto_three_lanes')
        self.declare_parameter('min_row_length', 1.5)
        self.declare_parameter('max_row_length', 100.0)
        self.declare_parameter('low_conf_frames_threshold', 15)
        self.declare_parameter('drive_out_distance', 1.80)
        self.declare_parameter('drive_out_max_time', 30.0)
        self.declare_parameter('min_turn_angle_deg', 140.0)
        self.declare_parameter('max_turn_angle_deg', 200.0)
        self.declare_parameter('reactive_avoid_wait_time', 3.0)  # seconds to wait before planning bypass
        self.declare_parameter('recovery_backup_distance', 1.0)  # meters to back up during recovery
        self.declare_parameter('use_hsv_mask', False)
        self.declare_parameter('datum_latitude', 10.775667)
        self.declare_parameter('datum_longitude', 106.670889)
        self.declare_parameter('datum_altitude', 10.0)
        self.declare_parameter('gps_topic', '/gps/fix')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('enable_depth_safety', True)
        self.declare_parameter('depth_estop_dist', 0.22)
        self.declare_parameter('enable_arrow_landmarks', True)
        self.declare_parameter('yolo_model_path', 'models/bao1.onnx')
        self.declare_parameter('yellow_arrow_turn_dist', 0.50)
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('imu_topic', '/imu/data')
        
        default_log_dir = os.path.join(
            os.path.expanduser('~'), 'ros2_telemetry_logs'
        )
        self.declare_parameter('enable_file_logging', True)
        self.declare_parameter('log_output_dir', default_log_dir)
        self.declare_parameter('terminal_log_interval', 1.0)
        self.declare_parameter('save_debug_imgs', False)

        # ── Tham số Điều Phối Nhiệm Vụ 2 Luống & Quỹ Đạo Omega Turn ───
        self.declare_parameter('enable_mission_goals', True)
        self.declare_parameter('enable_global_plan_assist', True)
        self.declare_parameter('field_length', 3.0)
        self.declare_parameter('headland_area_length', 4.0)
        self.declare_parameter('goal_tolerance', 0.40)
        self.declare_parameter('goal_1_x', 3.0)
        self.declare_parameter('goal_1_y', 0.0)
        self.declare_parameter('goal_2_x', 3.0)
        self.declare_parameter('goal_2_y', -1.20)
        self.declare_parameter('goal_3_x', 0.0)
        self.declare_parameter('goal_3_y', -1.20)
        self.declare_parameter('turn_side', 'RIGHT')
        self.declare_parameter('omega_open_angle_deg', 35.0)
        self.declare_parameter('omega_r1', 0.85)
        self.declare_parameter('omega_clearance', 0.55)
        self.declare_parameter('omega_lead_in', 0.50)
        self.declare_parameter('wait_for_rviz_start', False)

        # ── Ngưỡng an toàn LiDAR (đo từ tâm LiDAR, m) ──
        self.declare_parameter('robot_half_width', 0.34)      # Nửa bề rộng xe tới mép bánh ngoài
        self.declare_parameter('front_stop_dist', 0.22)       # Dừng khi vật cản trước mũi < ngưỡng
        self.declare_parameter('lidar_estop_dist', 0.20)      # Phanh phản xạ 15Hz trong scan_callback
        self.declare_parameter('side_near_dist', 0.42)        # Sát sườn: tăng độ nhạy lái + repulsion
        self.declare_parameter('side_critical_dist', 0.39)    # Rất sát: giảm 50% tốc độ
        self.declare_parameter('side_emergency_dist', 0.36)   # Tới hạn: bò chậm + lái gắt thoát sườn
        self.declare_parameter('side_emergency_speed_ratio', 0.40)  # Tỉ lệ linear_speed khi tới hạn

        p = self.get_parameter
        self.model_path               = p('model_path').value
        self.input_height             = p('input_height').value
        self.input_width              = p('input_width').value
        self.num_threads              = p('num_threads').value
        self.mask_threshold           = p('mask_threshold').value
        self.linear_speed             = p('linear_speed').value
        self.turn_linear_speed        = p('turn_linear_speed').value
        self.turn_angular_speed       = p('turn_angular_speed').value
        self.yellow_arrow_turn_angle_deg = float(p('yellow_arrow_turn_angle_deg').value)
        self.low_confidence_threshold = p('low_confidence_threshold').value
        self.high_confidence_threshold= p('high_confidence_threshold').value
        self.lambda_smc               = p('lambda_smc').value
        self.k_smc                    = p('k_smc').value
        self.eta_smc                  = p('eta_smc').value
        self.phi_smc                  = p('phi_smc').value
        self.max_steering_angle_deg   = p('max_steering_angle_deg').value
        self.turn_in_place_threshold_deg = float(p('turn_in_place_threshold_deg').value)
        self.turn_in_place_resume_deg    = float(p('turn_in_place_resume_deg').value)
        self.row_spacing              = p('row_spacing').value
        self.ema_alpha                = p('ema_alpha').value
        self.enable_uturn             = p('enable_uturn').value
        self.warmup_time              = p('warmup_time').value
        self.navigation_mode          = p('navigation_mode').value
        self.min_row_length           = p('min_row_length').value
        self.max_row_length           = p('max_row_length').value
        self.low_conf_frames_threshold= p('low_conf_frames_threshold').value
        self.drive_out_distance       = float(p('drive_out_distance').value)
        self.drive_out_max_time       = float(p('drive_out_max_time').value if self.has_parameter('drive_out_max_time') else 9.0)
        self.min_turn_angle_deg       = p('min_turn_angle_deg').value
        self.max_turn_angle_deg       = p('max_turn_angle_deg').value
        self.reactive_avoid_wait_time = p('reactive_avoid_wait_time').value
        self.recovery_backup_distance = p('recovery_backup_distance').value
        self.use_hsv_mask             = p('use_hsv_mask').value
        self.datum_latitude          = p('datum_latitude').value
        self.datum_longitude         = p('datum_longitude').value
        self.datum_altitude          = p('datum_altitude').value
        self.gps_topic               = p('gps_topic').value
        self.image_topic             = p('image_topic').value
        self.depth_topic             = p('depth_topic').value if self.has_parameter('depth_topic') else '/camera/depth/image_raw'
        self.enable_depth_safety     = bool(p('enable_depth_safety').value) if self.has_parameter('enable_depth_safety') else True
        self.depth_estop_dist        = float(p('depth_estop_dist').value) if self.has_parameter('depth_estop_dist') else 0.22
        self.odom_topic              = p('odom_topic').value
        self.imu_topic               = p('imu_topic').value
        self.enable_file_logging     = p('enable_file_logging').value
        self.log_output_dir          = p('log_output_dir').value
        self.terminal_log_interval   = p('terminal_log_interval').value
        self.save_debug_imgs         = p('save_debug_imgs').value
        self._last_terminal_log_time = 0.0

        # Mission goals & Omega Turn parameters
        self.enable_mission_goals     = p('enable_mission_goals').value
        self.enable_global_plan_assist= p('enable_global_plan_assist').value
        self.field_length             = p('field_length').value
        self.headland_area_length     = p('headland_area_length').value
        self.goal_tolerance           = p('goal_tolerance').value
        self.goal_1_x                 = p('goal_1_x').value
        self.goal_1_y                 = p('goal_1_y').value
        self.goal_2_x                 = p('goal_2_x').value
        self.goal_2_y                 = p('goal_2_y').value
        self.goal_3_x                 = p('goal_3_x').value
        self.goal_3_y                 = p('goal_3_y').value
        self.turn_side                = p('turn_side').value
        self.omega_open_angle_deg     = p('omega_open_angle_deg').value
        self.omega_r1                 = p('omega_r1').value
        self.omega_clearance          = p('omega_clearance').value
        self.omega_lead_in            = p('omega_lead_in').value
        self.wait_for_rviz_start      = p('wait_for_rviz_start').value

        self.robot_half_width         = float(p('robot_half_width').value)
        self.front_stop_dist          = float(p('front_stop_dist').value)
        self.lidar_estop_dist         = float(p('lidar_estop_dist').value)
        self.side_near_dist           = float(p('side_near_dist').value)
        self.side_critical_dist       = float(p('side_critical_dist').value)
        self.side_emergency_dist      = float(p('side_emergency_dist').value)
        self.side_emergency_speed_ratio = float(np.clip(p('side_emergency_speed_ratio').value, 0.0, 1.0))
        self._validate_safety_thresholds()
        self.is_mission_started       = not self.wait_for_rviz_start
        self.current_lane_idx         = 1

        if self.enable_file_logging:
            self.telemetry_logger = TelemetryLogger(log_dir=self.log_output_dir)
        else:
            self.telemetry_logger = None

        # ── CV Bridge ─────────────────────────────────────────────────
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().warn("CvBridge unavailable — using numpy fallback.")

        # ── Depth Sensor Variables (Intel RealSense D435) ─────────────
        self.front_depth_dist = float('inf')
        self._last_depth_time = 0.0
        self._latest_depth_raw = None

        # ── YOLO Arrow Landmark Detector (Điều kiện 1: Vàng <= 0.5m, Điều kiện 2: Đen <= 0.5m) ──
        self.enable_arrow_landmarks   = bool(p('enable_arrow_landmarks').value) if self.has_parameter('enable_arrow_landmarks') else True
        self.yolo_model_path          = str(p('yolo_model_path').value) if self.has_parameter('yolo_model_path') else 'models/bao1.onnx'
        self.yellow_arrow_turn_dist   = float(p('yellow_arrow_turn_dist').value) if self.has_parameter('yellow_arrow_turn_dist') else 0.50
        self.black_arrow_turn_dist    = float(p('black_arrow_turn_dist').value) if self.has_parameter('black_arrow_turn_dist') else 0.50
        self.arrow_detector           = YoloArrowDetector(model_path=self.yolo_model_path)
        self._latest_arrow_res        = None

        if not self.use_hsv_mask:
            if not self.model_path or not os.path.exists(self.model_path):
                target_name = os.path.basename(self.model_path) if self.model_path else 'crop_row_cnn_best.onnx'
                candidate_names = [target_name, 'crop_row_cnn_best.onnx', 'crop_row_cnn_best_final.onnx', 'crop_row_cnn_best_final_int8.onnx']
                
                # 0. Try workspace models directory
                ws_models = os.path.join(os.path.expanduser('~'), 'robot_ws', 'models')
                for name in candidate_names:
                    candidate = os.path.join(ws_models, name)
                    if os.path.exists(candidate):
                        self.model_path = candidate
                        break

                # 1. Try my_robot_controller share directory
                if not self.model_path or not os.path.exists(self.model_path):
                    try:
                        from ament_index_python.packages import get_package_share_directory
                        share_dir = get_package_share_directory('my_robot_controller')
                        for name in candidate_names:
                            candidate = os.path.join(share_dir, 'models', name)
                            if os.path.exists(candidate):
                                self.model_path = candidate
                                break
                    except Exception:
                        pass

                # 2. Try luanvan_control share directory
                if not self.model_path or not os.path.exists(self.model_path):
                    try:
                        from ament_index_python.packages import get_package_share_directory
                        share_dir = get_package_share_directory('luanvan_control')
                        for name in candidate_names:
                            candidate = os.path.join(share_dir, 'models', name)
                            if os.path.exists(candidate):
                                self.model_path = candidate
                                break
                    except Exception:
                        pass

                # 3. Fallback to relative path from source files
                if not self.model_path or not os.path.exists(self.model_path):
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    for name in candidate_names:
                        candidate = os.path.abspath(os.path.join(current_dir, '..', 'models', name))
                        if os.path.exists(candidate):
                            self.model_path = candidate
                            break

            if not self.model_path or not os.path.exists(self.model_path):
                self.get_logger().error(f"ONNX model not found anywhere: {self.model_path}")
                sys.exit(1)

        self.inference = InferenceHandler(
            model_path=self.model_path,
            mask_threshold=self.mask_threshold,
            input_size=(self.input_height, self.input_width),
            use_hsv_mask=self.use_hsv_mask,
            num_threads=self.num_threads
        )
        self.lidar_processor = LidarProcessor()
        self.eor_detector = EndOfRowDetector(
            min_row_distance=self.min_row_length,
            low_confidence_threshold=self.low_confidence_threshold,
            consecutive_frames=self.low_conf_frames_threshold
        )
        self.perception_manager = PerceptionManager(
            inference_handler=self.inference,
            lidar_processor=self.lidar_processor,
            eor_detector=self.eor_detector,
            front_stop_dist=self.front_stop_dist,
            side_danger_dist=self.side_near_dist
        )
        self.localization_manager = LocalizationManager(
            datum_lat=self.datum_latitude,
            datum_lon=self.datum_longitude,
            datum_alt=self.datum_altitude
        )

        # ── Decoupled Architecture Components ─────────────────────────
        self.fsm = FSMCoordinator(FSMState.TRACKING)

        self.tracking_controller = TrackingControllerSMC()
        self.tracking_controller.initialize(
            lambda_smc=self.lambda_smc,
            k_smc=self.k_smc,
            eta_smc=self.eta_smc,
            phi_smc=self.phi_smc,
            linear_speed=self.linear_speed,
            turn_angular_speed=self.turn_angular_speed
        )

        self.pure_pursuit_controller = PurePursuitController()
        # Đảm bảo tốc độ tiến bám cung quay đầu Omega (chuẩn 0.06 m/s) êm, đủ lực
        path_follow_speed = max(float(getattr(self, 'turn_linear_speed', 0.06)), 0.05)
        self.pure_pursuit_controller.initialize(
            turn_linear_speed=path_follow_speed,
            turn_angular_speed=self.turn_angular_speed
        )

        self.controller_manager = ControllerManager()
        self.controller_manager.register_controller('smc', self.tracking_controller)
        self.controller_manager.register_controller('pure_pursuit', self.pure_pursuit_controller)

        # ── FSM state variables ───────────────────────────────────────
        self.state_start_time       = None
        self.node_start_time        = None
        self.smoothed_angle_deg     = 0.0
        self.low_confidence_counter = 0
        self.high_confidence_counter = 0
        self.eor_low_conf_frames    = 0
        self.inside_row             = False
        self.has_seen_row           = False
        self.is_adjusting_heading   = False
        self.heading_adjust_target_yaw = 0.0
        self.heading_adjust_dir     = 0.0
        self.heading_adjust_start_time = 0.0
        self.heading_adjust_cooldown_until = 0.0
        self._aligned_frame_count   = 0
        self.current_lane_y         = 0.0

        # ── Odometry tracking ─────────────────────────────────────────
        self.distance_traveled  = 0.0    # m — cộng dồn từ đầu hàng
        self._prev_odom_x       = None
        self._prev_odom_y       = None
        self._odom_received     = False

        # New pose & orientation variables
        self.current_x          = 0.0
        self.current_y          = 0.0
        self.current_yaw        = 0.0

        # Drive out & turn tracking variables
        self.drive_out_start_x  = 0.0
        self.drive_out_start_y  = 0.0
        self.drive_out_start_dist = 0.0
        self.drive_out_start_time = 0.0
        self.drive_out_target_yaw = 0.0
        self._is_final_goal     = False
        self.rotate_start_yaw   = 0.0
        self.accumulated_turn_angle = 0.0
        self._prev_yaw_for_turn = None
        self.turn_direction     = 1.0  # 1.0 = left (CCW), -1.0 = right (CW)

        # ── Ghi nhớ góc IMU "tốt" trung bình khi CNN còn thấy rõ hàng ──
        # Khi xe đang bám luống ổn định (confidence >= 0.45), liên tục cập nhật
        # _good_row_yaw_ema để lưu lại hướng đi chuẩn chính xác nhất.
        # Khi hết nhìn thấy hàng -> dùng _good_row_yaw_ema thay vì góc IMU tức thời!
        self._good_row_yaw_ema   = 0.0
        self._good_row_yaw_valid = False

        # ── Bộ dừng quay nhanh theo IMU (Fast Turn Guard, chạy luồng riêng ~50Hz) ──
        # Vòng camera chỉ ~5-10 FPS khi chạy YOLO -> dừng quay 90° bằng camera bị trễ 1 khung (~5°).
        # Khi đang xoay 90° (Nhịp 1 & 3), callback IMU tự phát lệnh xoay và tự cắt lệnh ngay khi đủ góc.
        self._turn_lock            = threading.Lock()
        self._fast_turn_active     = False
        self._fast_turn_reached    = False
        self._fast_turn_target_yaw = 0.0
        self._fast_turn_sign       = -1.0
        self._last_imu_cb_time     = 0.0
        self._suppress_cmd_publish = False
        self._drive_out_ref_valid  = False   # drive_out_target_yaw vừa được chốt cho lượt Luống 1 -> 2
        self.uturn_row1_yaw        = 0.0     # Hướng chuẩn tim Luống 1 (tham chiếu tuyệt đối cho 2 lần quay 90°)
        self.uturn_turn1_target_yaw = 0.0    # Hướng lý tưởng sau Nhịp 1 (Luống 1 + 90°)
        self._last_black_seen_time = 0.0
        self._last_black_seen_yaw  = 0.0
        self.turn_stop_lead_sec    = 0.12    # s — bù trễ dừng (IMU + bridge + ESP32) theo tốc độ quay thực tế

        # Recovery tracking variables
        self.recovery_start_x   = 0.0
        self.recovery_start_y   = 0.0

        # Navigation tracking
        self.last_visited_lane  = None  # can be 'lane1', 'lane2', 'lane3'
        self.latest_scan        = None
        self.last_path_completion_time = -100.0
        self.row_start_x        = None
        self.row_completed      = False
        self.uturn_goal_x       = None
        self.uturn_goal_y       = None
        self.uturn_target_lane  = None

        # ── IMU U-Turn & Slip Monitoring Variables ────────────────────
        self.uturn_start_imu_yaw      = 0.0
        self.uturn_target_yaw         = 0.0
        self.uturn_accum_imu_yaw      = 0.0
        self.uturn_accum_wheel_yaw    = 0.0
        self._prev_imu_yaw_for_turn   = None
        self._prev_wheel_yaw_for_turn = None
        self._slip_warning_logged     = False
        self.is_trimming_uturn_heading= False
        self.uturn_trim_start_time    = 0.0

        # ── Image logging ─────────────────────────────────────────────
        self.last_img_save_time = 0.0
        self.output_dir = os.path.join(os.path.expanduser('~'), 'ros2_debug_imgs')

        # ── Publishers & Subscribers ──────────────────────────────────
        self.cmd_vel_pub   = self.create_publisher(Twist, 'cmd_vel', 10)
        self.gps_pub       = self.create_publisher(NavSatFix, '/localization/gps', 10)
        self.plan_pub      = self.create_publisher(RosPath, '/plan', 10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/mission_markers', 10)
        self.plan_timer    = self.create_timer(0.5, self._plan_and_markers_timer_callback)
        self.heading_adjust_timer = self.create_timer(0.05, self._heading_adjust_timer_callback)
        self.ai_view_pub   = self.create_publisher(Image, '/camera/ai_view', 2)
        self.mask_pub      = self.create_publisher(Image, '/crop_row/mask', 2)
        self._received_frame_count = 0

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)

        if self.enable_depth_safety:
            self.depth_sub = self.create_subscription(
                Image, self.depth_topic, self.depth_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)

        # Wheel odom subscriber (dùng so sánh góc quay encoder và IMU để phát hiện trượt bánh)
        self.wheel_odom_sub = self.create_subscription(
            Odometry, '/wheel/odom', self.wheel_odom_callback, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.gps_sub = self.create_subscription(
            NavSatFix, self.gps_topic, self.gps_callback, 10)

        # IMU chạy trong nhóm callback riêng (luồng riêng) để Fast Turn Guard phản ứng ~50Hz
        # ngay cả khi image_callback đang bận suy luận YOLO/CNN (onnxruntime nhả GIL khi chạy).
        self._imu_cb_group = MutuallyExclusiveCallbackGroup()
        self.imu_sub = self.create_subscription(
            Imu, self.imu_topic, self.imu_callback, 10, callback_group=self._imu_cb_group)
        # Fallback subscription for alternative imu topic
        if self.imu_topic not in ('/imu', 'imu'):
            self.imu_fallback_sub = self.create_subscription(
                Imu, '/imu', self.imu_callback, 10, callback_group=self._imu_cb_group)

        # ── Remote CNN Offloading (Laptop gánh phần suy luận ONNX) ───────
        # QoS BEST_EFFORT + depth=1: bỏ gói cũ, chỉ dùng kết quả mới nhất
        # → Khớp với publisher trên laptop (cnn_inference_server.py)
        # → Giảm trễ tối đa, tránh hàng đợi tích lũy
        _qos_realtime = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._remote_cnn_data = None
        self._remote_cnn_time = 0.0
        self.crop_row_sub = self.create_subscription(
            Float32MultiArray, '/crop_row/detection', self.remote_cnn_callback, _qos_realtime
        )

        # Cho phép người dùng tự chấm điểm Goal trực tiếp trên RViz bằng công cụ 2D Goal Pose
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10
        )
        # Cho phép kích hoạt xuất phát từ topic /start_mission
        self.start_mission_sub = self.create_subscription(
            Empty, '/start_mission', self.start_mission_callback, 10
        )

        # ── Đo lường vận tốc Doppler GPS NEO-M10 & Đánh giá Omega Turn ───
        self.latest_gps_speed = None
        self.gps_vel_sub = self.create_subscription(
            TwistStamped, '/gps/vel', self.gps_vel_callback, 10
        )
        self.omega_evaluator = OmegaTurnEvaluator()

        self.get_logger().info(
            f"🚀 [AI CNN Driver] Sẵn sàng tự hành bám luống (Vận tốc: {self.linear_speed:.2f} m/s | Model INT8)"
        )

        # ── State for Periodic 1Hz Status Logging ─────────────────────
        self._last_image_time = 0.0
        self._waiting_camera_notified = False
        self._camera_lost_notified = False
        self._latest_confidence = 0.0
        self._latest_steer_deg = 0.0
        self._latest_twist = Twist()
        self._latest_gps_info = {}
        self._latest_inf_ms = 0.0
        self.status_timer = self.create_timer(1.0, self.status_timer_callback)

    def gps_vel_callback(self, msg: TwistStamped):
        """Tiếp nhận vận tốc Doppler tức thời từ GPS NEO-M10 (/gps/vel)."""
        self.latest_gps_speed = msg.twist.linear.x

    def start_mission_callback(self, msg: Empty):
        """Nhận tín hiệu kích hoạt xuất phát từ topic /start_mission."""
        if not self.is_mission_started:
            self.is_mission_started = True
            now = self.get_clock().now()
            self.node_start_time = now
            self.state_start_time = now
            self.get_logger().info("🚀 [XUẤT PHÁT] Đã nhận lệnh xuất phát từ /start_mission! Xe bắt đầu lăn bánh...")

    def goal_pose_callback(self, msg: PoseStamped):
        """Nhận điểm Goal do người dùng tự chấm bằng chuột trên RViz (nút 2D Goal Pose)."""
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)

        # Nếu xe đang ở trạng thái chờ (READY), cú click 2D Goal Pose trên RViz sẽ lập tức kích hoạt xe xuất phát!
        if not self.is_mission_started:
            self.is_mission_started = True
            now = self.get_clock().now()
            self.node_start_time = now
            self.state_start_time = now
            cam_lost = (self._last_image_time == 0.0) or ((time.time() - self._last_image_time) > 2.5)
            if cam_lost:
                self.get_logger().warn(
                    f"⚠️ [XUẤT PHÁT TỪ RVIZ] Đã nhận lệnh từ 2D Goal Pose ({gx:.2f}m, {gy:.2f}m), "
                    f"NHƯNG CAMERA ĐANG MẤT TÍN HIỆU! Xe tạm dừng chờ Camera kết nối lại để bám luống..."
                )
            else:
                self.get_logger().info(
                    f"🚀 [XUẤT PHÁT TỪ RVIZ] Đã nhận lệnh điều khiển từ 2D Goal Pose ({gx:.2f}m, {gy:.2f}m)! Xe bắt đầu lăn bánh..."
                )

        if self.current_lane_idx == 1:
            if gx > 1.0:
                self.goal_1_x = gx
                self.goal_1_y = gy
                self.field_length = max(1.0, gx)
                self.goal_2_x = gx
                self.goal_2_y = gy - abs(self.row_spacing)
                self.get_logger().info(
                    f"🎯 [RVIZ 2D GOAL] Đã cập nhật GOAL 1 theo điểm bạn vừa chấm: ({gx:.2f}m, {gy:.2f}m). "
                    f"Chiều dài luống mới: {self.field_length:.2f}m."
                )
        elif self.current_lane_idx == 2:
            self.goal_3_x = gx
            self.goal_3_y = gy
            self.get_logger().info(
                f"🎯 [RVIZ 2D GOAL] Đã cập nhật GOAL 3 (Đích đến) theo điểm bạn vừa chấm: ({gx:.2f}m, {gy:.2f}m)!"
            )

    # ── FSM Coordination Interface Methods ───────────────────────────

    def StartTracking(self, dt):
        """Enable tracking controller and calculate output."""
        self.controller_manager.select_controller('smc')
        res = self.controller_manager.compute_command(self.smoothed_angle_deg, dt_actual=dt)
        return res["linear_velocity"], res["angular_velocity"]

    def FollowPath(self):
        """Enable path following controller and calculate output."""
        self.controller_manager.select_controller('pure_pursuit')
        res = self.controller_manager.compute_command(self.current_x, self.current_y, self.current_yaw)
        finished = (res["status"] == "COMPLETED")
        return res["linear_velocity"], res["angular_velocity"], finished, self.pure_pursuit_controller.path_index

    def _plan_and_markers_timer_callback(self):
        """Timer chu kỳ 2Hz xuất bản lộ trình toàn cục và các cọc tiêu Marker chấm điểm."""
        self.publish_mission_path()
        self.publish_mission_markers()

    def publish_mission_path(self):
        """Xuất bản toàn bộ lộ trình nhiệm vụ 2 luống + Omega Turn lên /plan cho RViz hiển thị trực quan."""
        try:
            full_mission = OmegaTurnPlanner.generate_full_mission_path(
                field_length=self.field_length,
                row_spacing=self.row_spacing,
                turn_side=self.turn_side,
                open_angle_deg=self.omega_open_angle_deg,
                r1=self.omega_r1,
                clearance_dist=self.omega_clearance,
                lead_in_dist=self.omega_lead_in,
            )
            now_msg = self.get_clock().now().to_msg()
            ros_path = OmegaTurnPlanner.to_ros_path(full_mission.waypoints, frame_id='odom', stamp=now_msg)
            self.plan_pub.publish(ros_path)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish mission path: {e}", throttle_duration_sec=5.0)

    def publish_mission_markers(self):
        """
        Xuất bản trực quan hóa các điểm Goal, luống cây quét thực tế và bảng chấm điểm
        tiến độ nhiệm vụ lên RViz qua topic /mission_markers.
        """
        try:
            marker_arr = MarkerArray()
            now_msg = self.get_clock().now().to_msg()
            frame_id = 'odom'

            # ── 1. Cọc tiêu START (0, 0) ──────────────────────────────
            start_cyl = Marker()
            start_cyl.header.frame_id = frame_id
            start_cyl.header.stamp = now_msg
            start_cyl.ns = "mission_goals"
            start_cyl.id = 10
            start_cyl.type = Marker.CYLINDER
            start_cyl.action = Marker.ADD
            start_cyl.pose.position.x = 0.0
            start_cyl.pose.position.y = 0.0
            start_cyl.pose.position.z = 0.20
            start_cyl.pose.orientation.w = 1.0
            start_cyl.scale.x = 0.25
            start_cyl.scale.y = 0.25
            start_cyl.scale.z = 0.40
            start_cyl.color.r = 0.0
            start_cyl.color.g = 0.8
            start_cyl.color.b = 1.0
            start_cyl.color.a = 0.85
            marker_arr.markers.append(start_cyl)

            start_txt = Marker()
            start_txt.header.frame_id = frame_id
            start_txt.header.stamp = now_msg
            start_txt.ns = "mission_goals"
            start_txt.id = 11
            start_txt.type = Marker.TEXT_VIEW_FACING
            start_txt.action = Marker.ADD
            start_txt.pose.position.x = 0.0
            start_txt.pose.position.y = 0.0
            start_txt.pose.position.z = 0.55
            start_txt.pose.orientation.w = 1.0
            start_txt.scale.z = 0.22
            start_txt.color.r = 0.0
            start_txt.color.g = 0.9
            start_txt.color.b = 1.0
            start_txt.color.a = 1.0
            start_txt.text = "START (0, 0)\nDau Luong 1"
            marker_arr.markers.append(start_txt)

            # ── 2. Cọc tiêu GOAL 1 (Cuối Luống 1) ─────────────────────
            g1_cyl = Marker()
            g1_cyl.header.frame_id = frame_id
            g1_cyl.header.stamp = now_msg
            g1_cyl.ns = "mission_goals"
            g1_cyl.id = 20
            g1_cyl.type = Marker.CYLINDER
            g1_cyl.action = Marker.ADD
            g1_cyl.pose.position.x = float(self.goal_1_x)
            g1_cyl.pose.position.y = float(self.goal_1_y)
            g1_cyl.pose.position.z = 0.25
            g1_cyl.pose.orientation.w = 1.0
            g1_cyl.scale.x = 0.30
            g1_cyl.scale.y = 0.30
            g1_cyl.scale.z = 0.50
            g1_cyl.color.r = 1.0
            g1_cyl.color.g = 0.85
            g1_cyl.color.b = 0.0
            g1_cyl.color.a = 0.90
            marker_arr.markers.append(g1_cyl)

            g1_txt = Marker()
            g1_txt.header.frame_id = frame_id
            g1_txt.header.stamp = now_msg
            g1_txt.ns = "mission_goals"
            g1_txt.id = 21
            g1_txt.type = Marker.TEXT_VIEW_FACING
            g1_txt.action = Marker.ADD
            g1_txt.pose.position.x = float(self.goal_1_x)
            g1_txt.pose.position.y = float(self.goal_1_y)
            g1_txt.pose.position.z = 0.70
            g1_txt.pose.orientation.w = 1.0
            g1_txt.scale.z = 0.24
            g1_txt.color.r = 1.0
            g1_txt.color.g = 0.9
            g1_txt.color.b = 0.0
            g1_txt.color.a = 1.0
            g1_txt.text = f"GOAL 1 ({self.goal_1_x:.1f}, {self.goal_1_y:.1f})\nCuoi Luong 1"
            marker_arr.markers.append(g1_txt)

            # ── 3. Cọc tiêu GOAL 2 (Đầu Luống 2) ─────────────────────
            g2_cyl = Marker()
            g2_cyl.header.frame_id = frame_id
            g2_cyl.header.stamp = now_msg
            g2_cyl.ns = "mission_goals"
            g2_cyl.id = 30
            g2_cyl.type = Marker.CYLINDER
            g2_cyl.action = Marker.ADD
            g2_cyl.pose.position.x = float(self.goal_2_x)
            g2_cyl.pose.position.y = float(self.goal_2_y)
            g2_cyl.pose.position.z = 0.25
            g2_cyl.pose.orientation.w = 1.0
            g2_cyl.scale.x = 0.30
            g2_cyl.scale.y = 0.30
            g2_cyl.scale.z = 0.50
            g2_cyl.color.r = 1.0
            g2_cyl.color.g = 0.2
            g2_cyl.color.b = 1.0
            g2_cyl.color.a = 0.90
            marker_arr.markers.append(g2_cyl)

            g2_txt = Marker()
            g2_txt.header.frame_id = frame_id
            g2_txt.header.stamp = now_msg
            g2_txt.ns = "mission_goals"
            g2_txt.id = 31
            g2_txt.type = Marker.TEXT_VIEW_FACING
            g2_txt.action = Marker.ADD
            g2_txt.pose.position.x = float(self.goal_2_x)
            g2_txt.pose.position.y = float(self.goal_2_y)
            g2_txt.pose.position.z = 0.70
            g2_txt.pose.orientation.w = 1.0
            g2_txt.scale.z = 0.24
            g2_txt.color.r = 1.0
            g2_txt.color.g = 0.3
            g2_txt.color.b = 1.0
            g2_txt.color.a = 1.0
            g2_txt.text = f"GOAL 2 ({self.goal_2_x:.1f}, {self.goal_2_y:.1f})\nDau Luong 2"
            marker_arr.markers.append(g2_txt)

            # ── 4. Cọc tiêu GOAL 3 (Đích đến kết thúc nhiệm vụ) ────────
            g3_cyl = Marker()
            g3_cyl.header.frame_id = frame_id
            g3_cyl.header.stamp = now_msg
            g3_cyl.ns = "mission_goals"
            g3_cyl.id = 40
            g3_cyl.type = Marker.CYLINDER
            g3_cyl.action = Marker.ADD
            g3_cyl.pose.position.x = float(self.goal_3_x)
            g3_cyl.pose.position.y = float(self.goal_3_y)
            g3_cyl.pose.position.z = 0.30
            g3_cyl.pose.orientation.w = 1.0
            g3_cyl.scale.x = 0.35
            g3_cyl.scale.y = 0.35
            g3_cyl.scale.z = 0.60
            g3_cyl.color.r = 0.1
            g3_cyl.color.g = 1.0
            g3_cyl.color.b = 0.2
            g3_cyl.color.a = 0.95
            marker_arr.markers.append(g3_cyl)

            g3_txt = Marker()
            g3_txt.header.frame_id = frame_id
            g3_txt.header.stamp = now_msg
            g3_txt.ns = "mission_goals"
            g3_txt.id = 41
            g3_txt.type = Marker.TEXT_VIEW_FACING
            g3_txt.action = Marker.ADD
            g3_txt.pose.position.x = float(self.goal_3_x)
            g3_txt.pose.position.y = float(self.goal_3_y)
            g3_txt.pose.position.z = 0.80
            g3_txt.pose.orientation.w = 1.0
            g3_txt.scale.z = 0.26
            g3_txt.color.r = 0.1
            g3_txt.color.g = 1.0
            g3_txt.color.b = 0.3
            g3_txt.color.a = 1.0
            g3_txt.text = f"GOAL 3 ({self.goal_3_x:.1f}, {self.goal_3_y:.1f})\nDICH DEN (FINISH)"
            marker_arr.markers.append(g3_txt)

            # ── 5. Bố trí thực tế: 12 thùng carton (3 hàng x 4 thùng/hàng, 22x20cm cao 65cm) ──
            spacing = abs(self.row_spacing)
            row_y_list = [0.5 * spacing, -0.5 * spacing, -1.5 * spacing]
            L = float(self.field_length)

            boxes_marker = Marker()
            boxes_marker.header.frame_id = frame_id
            boxes_marker.header.stamp = now_msg
            boxes_marker.ns = "carton_boxes"
            boxes_marker.id = 50
            boxes_marker.type = Marker.CUBE_LIST
            boxes_marker.action = Marker.ADD
            boxes_marker.pose.orientation.w = 1.0
            # Kích thước thùng thật: dài 22cm, rộng 20cm, cao 65cm
            boxes_marker.scale.x = 0.22
            boxes_marker.scale.y = 0.20
            boxes_marker.scale.z = 0.65
            # Màu nâu carton thực tế
            boxes_marker.color.r = 0.82
            boxes_marker.color.g = 0.58
            boxes_marker.color.b = 0.32
            boxes_marker.color.a = 0.95

            # Mỗi hàng có đúng 4 thùng phân bố trên chiều dài L (x = 0, 1m, 2m, 3m)
            box_x_positions = [0.0, L / 3.0, 2.0 * L / 3.0, L]

            for ry in row_y_list:
                for x_val in box_x_positions:
                    p = Point()
                    p.x = float(x_val)
                    p.y = float(ry)
                    p.z = 0.325  # Đặt đáy thùng tiếp đất (cao 0.65m / 2)
                    boxes_marker.points.append(p)
            marker_arr.markers.append(boxes_marker)

            # Đường tim luống 1 và luống 2 (LINE_LIST)
            lanes_marker = Marker()
            lanes_marker.header.frame_id = frame_id
            lanes_marker.header.stamp = now_msg
            lanes_marker.ns = "crop_rows"
            lanes_marker.id = 51
            lanes_marker.type = Marker.LINE_LIST
            lanes_marker.action = Marker.ADD
            lanes_marker.pose.orientation.w = 1.0
            lanes_marker.scale.x = 0.04
            lanes_marker.color.r = 0.4
            lanes_marker.color.g = 0.7
            lanes_marker.color.b = 1.0
            lanes_marker.color.a = 0.75

            # Tim luống 1 (y = 0.0)
            p1_start = Point(x=0.0, y=0.0, z=0.02)
            p1_end = Point(x=L, y=0.0, z=0.02)
            lanes_marker.points.extend([p1_start, p1_end])

            # Tim luống 2 (y = -spacing)
            p2_start = Point(x=L, y=-spacing, z=0.02)
            p2_end = Point(x=0.0, y=-spacing, z=0.02)
            lanes_marker.points.extend([p2_start, p2_end])
            marker_arr.markers.append(lanes_marker)

            # ── 6. Đường cong quay đầu Omega & Các điểm mốc Waypoints ──
            omega_path = OmegaTurnPlanner.generate_path(
                start_x=float(self.goal_1_x),
                start_y=float(self.goal_1_y),
                start_yaw=0.0,
                row_spacing=self.row_spacing,
                turn_side=self.turn_side,
                open_angle_deg=self.omega_open_angle_deg,
                r1=self.omega_r1,
                clearance_dist=self.omega_clearance,
                lead_in_dist=self.omega_lead_in,
            )
            if omega_path and omega_path.waypoints:
                # 6.1. Dải đường cong liên tục (LINE_STRIP màu cam)
                omega_marker = Marker()
                omega_marker.header.frame_id = frame_id
                omega_marker.header.stamp = now_msg
                omega_marker.ns = "omega_turn"
                omega_marker.id = 60
                omega_marker.type = Marker.LINE_STRIP
                omega_marker.action = Marker.ADD
                omega_marker.pose.orientation.w = 1.0
                omega_marker.scale.x = 0.05
                omega_marker.color.r = 1.0
                omega_marker.color.g = 0.55
                omega_marker.color.b = 0.0
                omega_marker.color.a = 0.95
                for wp in omega_path.waypoints:
                    p = Point(x=float(wp[0]), y=float(wp[1]), z=0.03)
                    omega_marker.points.append(p)
                marker_arr.markers.append(omega_marker)

                # 6.2. Chuỗi các điểm mốc nhỏ liên tiếp (SPHERE_LIST cách nhau ~15cm)
                wp_spheres = Marker()
                wp_spheres.header.frame_id = frame_id
                wp_spheres.header.stamp = now_msg
                wp_spheres.ns = "omega_turn"
                wp_spheres.id = 61
                wp_spheres.type = Marker.SPHERE_LIST
                wp_spheres.action = Marker.ADD
                wp_spheres.pose.orientation.w = 1.0
                wp_spheres.scale.x = 0.10
                wp_spheres.scale.y = 0.10
                wp_spheres.scale.z = 0.10
                wp_spheres.color.r = 1.0
                wp_spheres.color.g = 0.80
                wp_spheres.color.b = 0.10
                wp_spheres.color.a = 0.90

                for wp in omega_path.waypoints[::3]:
                    p = Point(x=float(wp[0]), y=float(wp[1]), z=0.04)
                    wp_spheres.points.append(p)
                marker_arr.markers.append(wp_spheres)

                # 6.3. Ba điểm mốc then chốt (Milestones) trên cung quay đầu:
                pts = omega_path.waypoints
                # Mốc 1: Thoát luống (Lead-out)
                lead_out_idx = min(len(pts) - 1, max(1, int(self.omega_clearance / 0.05)))
                p_lead_out = pts[lead_out_idx]

                m1_cyl = Marker()
                m1_cyl.header.frame_id = frame_id
                m1_cyl.header.stamp = now_msg
                m1_cyl.ns = "omega_turn"
                m1_cyl.id = 62
                m1_cyl.type = Marker.CYLINDER
                m1_cyl.action = Marker.ADD
                m1_cyl.pose.position.x = float(p_lead_out[0])
                m1_cyl.pose.position.y = float(p_lead_out[1])
                m1_cyl.pose.position.z = 0.15
                m1_cyl.pose.orientation.w = 1.0
                m1_cyl.scale.x = 0.18
                m1_cyl.scale.y = 0.18
                m1_cyl.scale.z = 0.30
                m1_cyl.color.r = 1.0
                m1_cyl.color.g = 0.9
                m1_cyl.color.b = 0.0
                m1_cyl.color.a = 0.90
                marker_arr.markers.append(m1_cyl)

                m1_txt = Marker()
                m1_txt.header.frame_id = frame_id
                m1_txt.header.stamp = now_msg
                m1_txt.ns = "omega_turn"
                m1_txt.id = 63
                m1_txt.type = Marker.TEXT_VIEW_FACING
                m1_txt.action = Marker.ADD
                m1_txt.pose.position.x = float(p_lead_out[0])
                m1_txt.pose.position.y = float(p_lead_out[1])
                m1_txt.pose.position.z = 0.40
                m1_txt.pose.orientation.w = 1.0
                m1_txt.scale.z = 0.18
                m1_txt.color.r = 1.0
                m1_txt.color.g = 1.0
                m1_txt.color.b = 0.2
                m1_txt.color.a = 1.0
                m1_txt.text = "1. Thoat Luong"
                marker_arr.markers.append(m1_txt)

                # Mốc 2: Đỉnh cua xa nhất (Apex)
                apex_pt = max(pts, key=lambda pt: pt[0])
                m2_cyl = Marker()
                m2_cyl.header.frame_id = frame_id
                m2_cyl.header.stamp = now_msg
                m2_cyl.ns = "omega_turn"
                m2_cyl.id = 64
                m2_cyl.type = Marker.CYLINDER
                m2_cyl.action = Marker.ADD
                m2_cyl.pose.position.x = float(apex_pt[0])
                m2_cyl.pose.position.y = float(apex_pt[1])
                m2_cyl.pose.position.z = 0.18
                m2_cyl.pose.orientation.w = 1.0
                m2_cyl.scale.x = 0.22
                m2_cyl.scale.y = 0.22
                m2_cyl.scale.z = 0.36
                m2_cyl.color.r = 1.0
                m2_cyl.color.g = 0.4
                m2_cyl.color.b = 0.0
                m2_cyl.color.a = 0.95
                marker_arr.markers.append(m2_cyl)

                m2_txt = Marker()
                m2_txt.header.frame_id = frame_id
                m2_txt.header.stamp = now_msg
                m2_txt.ns = "omega_turn"
                m2_txt.id = 65
                m2_txt.type = Marker.TEXT_VIEW_FACING
                m2_txt.action = Marker.ADD
                m2_txt.pose.position.x = float(apex_pt[0])
                m2_txt.pose.position.y = float(apex_pt[1])
                m2_txt.pose.position.z = 0.50
                m2_txt.pose.orientation.w = 1.0
                m2_txt.scale.z = 0.20
                m2_txt.color.r = 1.0
                m2_txt.color.g = 0.5
                m2_txt.color.b = 0.0
                m2_txt.color.a = 1.0
                m2_txt.text = "2. Dinh Cua (Apex)"
                marker_arr.markers.append(m2_txt)

                # Mốc 3: Nắn thẳng vào Luống 2 (Lead-in)
                lead_in_idx = max(0, len(pts) - 1 - max(1, int(self.omega_lead_in / 0.05)))
                p_lead_in = pts[lead_in_idx]

                m3_cyl = Marker()
                m3_cyl.header.frame_id = frame_id
                m3_cyl.header.stamp = now_msg
                m3_cyl.ns = "omega_turn"
                m3_cyl.id = 66
                m3_cyl.type = Marker.CYLINDER
                m3_cyl.action = Marker.ADD
                m3_cyl.pose.position.x = float(p_lead_in[0])
                m3_cyl.pose.position.y = float(p_lead_in[1])
                m3_cyl.pose.position.z = 0.15
                m3_cyl.pose.orientation.w = 1.0
                m3_cyl.scale.x = 0.18
                m3_cyl.scale.y = 0.18
                m3_cyl.scale.z = 0.30
                m3_cyl.color.r = 0.8
                m3_cyl.color.g = 0.2
                m3_cyl.color.b = 1.0
                m3_cyl.color.a = 0.90
                marker_arr.markers.append(m3_cyl)

                m3_txt = Marker()
                m3_txt.header.frame_id = frame_id
                m3_txt.header.stamp = now_msg
                m3_txt.ns = "omega_turn"
                m3_txt.id = 67
                m3_txt.type = Marker.TEXT_VIEW_FACING
                m3_txt.action = Marker.ADD
                m3_txt.pose.position.x = float(p_lead_in[0])
                m3_txt.pose.position.y = float(p_lead_in[1])
                m3_txt.pose.position.z = 0.40
                m3_txt.pose.orientation.w = 1.0
                m3_txt.scale.z = 0.18
                m3_txt.color.r = 0.9
                m3_txt.color.g = 0.3
                m3_txt.color.b = 1.0
                m3_txt.color.a = 1.0
                m3_txt.text = "3. Vao Luong 2"
                marker_arr.markers.append(m3_txt)

                # 6.4. Điểm mục tiêu nhìn trước thời gian thực (Lookahead Carrot Point)
                if getattr(self.pure_pursuit_controller, 'enabled', False) and getattr(self.pure_pursuit_controller, 'path', None):
                    c_path = self.pure_pursuit_controller.path
                    c_idx = min(len(c_path) - 1, self.pure_pursuit_controller.path_index + 4)
                    c_pt = c_path[c_idx]

                    carrot_marker = Marker()
                    carrot_marker.header.frame_id = frame_id
                    carrot_marker.header.stamp = now_msg
                    carrot_marker.ns = "omega_turn"
                    carrot_marker.id = 68
                    carrot_marker.type = Marker.SPHERE
                    carrot_marker.action = Marker.ADD
                    carrot_marker.pose.position.x = float(c_pt[0])
                    carrot_marker.pose.position.y = float(c_pt[1])
                    carrot_marker.pose.position.z = 0.20
                    carrot_marker.pose.orientation.w = 1.0
                    carrot_marker.scale.x = 0.20
                    carrot_marker.scale.y = 0.20
                    carrot_marker.scale.z = 0.20
                    carrot_marker.color.r = 1.0
                    carrot_marker.color.g = 0.1
                    carrot_marker.color.b = 0.1
                    carrot_marker.color.a = 0.95
                    marker_arr.markers.append(carrot_marker)

            # ── 7. Bảng chấm điểm & Tiến độ nhiệm vụ (HUD Marker) ──────
            cur_state = self.fsm.get_state()
            is_uturn = (cur_state == FSMState.UTURN_PLANNING or
                        (cur_state == FSMState.PATH_FOLLOWING and self.fsm.state_before_planning == FSMState.UTURN_PLANNING))

            p1_prog = 0.0
            p2_prog = 0.0
            p3_prog = 0.0
            total_score = 0.0
            status_text = ""

            if self.current_lane_idx == 1 and not is_uturn:
                p1_prog = min(100.0, max(0.0, (self.current_x / max(0.1, L)) * 100.0))
                total_score = p1_prog * 0.40
                status_text = "BAM LUONG 1 (CNN)"
            elif is_uturn:
                p1_prog = 100.0
                accum_deg = math.degrees(abs(getattr(self, 'uturn_accum_imu_yaw', 0.0)))
                p2_prog = min(100.0, (accum_deg / 180.0) * 100.0)
                total_score = 40.0 + p2_prog * 0.30
                status_text = f"QUAY DAU OMEGA ({accum_deg:.0f} deg/180 deg)"
            elif self.current_lane_idx == 2:
                p1_prog = 100.0
                p2_prog = 100.0
                run_l2 = max(0.0, L - self.current_x)
                p3_prog = min(100.0, max(0.0, (run_l2 / max(0.1, L)) * 100.0))
                total_score = 70.0 + p3_prog * 0.30
                if cur_state == FSMState.IDLE:
                    p3_prog = 100.0
                    total_score = 100.0
                    status_text = "DA VE DICH (HOAN THANH)!"
                else:
                    status_text = "BAM LUONG 2 (CNN)"

            hud_marker = Marker()
            hud_marker.header.frame_id = frame_id
            hud_marker.header.stamp = now_msg
            hud_marker.ns = "mission_hud"
            hud_marker.id = 1
            hud_marker.type = Marker.TEXT_VIEW_FACING
            hud_marker.action = Marker.ADD
            hud_marker.pose.position.x = float(self.current_x)
            hud_marker.pose.position.y = float(self.current_y)
            hud_marker.pose.position.z = 1.35  # Nâng cao lên 1.35m để không che khuất cọc xe
            hud_marker.pose.orientation.w = 1.0
            hud_marker.scale.z = 0.22
            cam_lost = (self._last_image_time == 0.0) or ((time.time() - self._last_image_time) > 2.5)
            if cam_lost:
                hud_marker.color.r = 1.0
                hud_marker.color.g = 0.15
                hud_marker.color.b = 0.15
                hud_marker.text = (
                    f"⚠️ [CANH BAO: CHUA CO TIN HIEU CAMERA!] ⚠️\n"
                    f"Trang thai: TAM DUNG AN TOAN (Cho hinh anh tu Camera)\n"
                    f"Vi tri: ({self.current_x:.2f}m, {self.current_y:.2f}m)\n"
                    f"👉 Kiem tra: Bat DroidCam tren dien thoai hoac cam lai Webcam USB!"
                )
            elif not self.is_mission_started:
                hud_marker.color.r = 1.0
                hud_marker.color.g = 1.0
                hud_marker.color.b = 0.2
                hud_marker.text = (
                    f"★ [SAN SANG (READY) - DANG CHO LENH RVIZ] ★\n"
                    f"Trang thai: DUNG YEN CHO LENH (Cam bien & AI da san sang)\n"
                    f"Vi tri: ({self.current_x:.2f}m, {self.current_y:.2f}m) | Goc: {self.smoothed_angle_deg:+.1f} deg\n"
                    f"👉 BAM NUT '2D Goal Pose' TREN RVIZ DE XE BAT DAU CHAY!"
                )
            else:
                hud_marker.color.r = 0.2
                hud_marker.color.g = 1.0
                hud_marker.color.b = 0.4
                eval_hud = ""
                if hasattr(self, 'omega_evaluator') and self.omega_evaluator.latest_result:
                    eval_hud = f"\n{self.omega_evaluator.get_hud_string()}"
                hud_marker.text = (
                    f"[TIEN DO NHIEM VU: {total_score:.1f}/100 DIEM]\n"
                    f"Trang thai: {status_text}\n"
                    f"Vi tri: ({self.current_x:.2f}m, {self.current_y:.2f}m) | Goc: {self.smoothed_angle_deg:+.1f} deg\n"
                    f"Tien do: L1: {p1_prog:.0f}% -> Quay: {p2_prog:.0f}% -> L2: {p3_prog:.0f}%"
                    f"{eval_hud}"
                )
            marker_arr.markers.append(hud_marker)

            self.marker_pub.publish(marker_arr)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish mission markers: {e}", throttle_duration_sec=5.0)

    def StopRobot(self):
        """Issue zero velocities to the actuators."""
        self.controller_manager.reset()
        if self.controller_manager.active_name in self.controller_manager.controllers:
            self.controller_manager.controllers[self.controller_manager.active_name].stop()
        self.controller_manager.active_name = None
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        return twist

    def Recovery(self):
        """Perform recovery backup maneuver."""
        # Calculate backup distance
        dx = self.current_x - self.recovery_start_x
        dy = self.current_y - self.recovery_start_y
        dist = math.sqrt(dx*dx + dy*dy)

        twist = Twist()
        if dist < self.recovery_backup_distance:
            twist.linear.x = -self.linear_speed  # Back up slowly
            twist.angular.z = 0.0
            finished = False
        else:
            finished = True
        return twist, finished

    # ── GPS, IMU & Wheel Odom callbacks ───────────────────────────────
    def wheel_odom_callback(self, msg: Odometry):
        """Theo dõi góc quay encoder bánh xe để so sánh với IMU phát hiện trượt bánh."""
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        wheel_yaw = math.atan2(siny, cosy)

        is_uturn_active = (
            self.fsm.get_state() == FSMState.UTURN_PLANNING or
            (self.fsm.get_state() == FSMState.PATH_FOLLOWING and self.fsm.state_before_planning == FSMState.UTURN_PLANNING) or
            self.is_trimming_uturn_heading
        )
        if is_uturn_active:
            if self._prev_wheel_yaw_for_turn is not None:
                dyaw_w = math.atan2(
                    math.sin(wheel_yaw - self._prev_wheel_yaw_for_turn),
                    math.cos(wheel_yaw - self._prev_wheel_yaw_for_turn)
                )
                self.uturn_accum_wheel_yaw += abs(dyaw_w)
            self._prev_wheel_yaw_for_turn = wheel_yaw
        else:
            self._prev_wheel_yaw_for_turn = None

    # ------------------------------------------------------------------
    # FAST TURN GUARD — điều khiển xoay 90° theo IMU ở tần số cao (luồng IMU riêng)
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _turn_remaining_deg(self, current_imu_yaw):
        """Góc còn phải quay (độ) theo đúng chiều quay. Âm = đã quay lố đích."""
        err = self._wrap_angle(self._fast_turn_target_yaw - current_imu_yaw)
        return math.degrees(err) * self._fast_turn_sign

    def _turn_rate_for_remaining(self, rem_deg, gyro_dps, phase_elapsed):
        """Tốc độ xoay theo góc còn lại: mạnh khi còn xa, hạ dần ở 25° cuối để dừng đúng 90° không lố."""
        base = float(self.turn_angular_speed)
        if rem_deg > 25.0:
            rate = base
            # Bù lực khi cỏ cản (gyro thực < 55% lệnh) — chỉ khi còn xa đích để không văng lố
            if phase_elapsed > 0.6 and gyro_dps < 0.55 * math.degrees(rate):
                rate = min(0.56, rate * 1.25)
        elif rem_deg > 10.0:
            rate = max(0.30, base * 0.72)
        else:
            rate = 0.28
        return rate

    def _arm_fast_turn(self, target_yaw, turn_sign):
        with self._turn_lock:
            self._fast_turn_target_yaw = float(target_yaw)
            self._fast_turn_sign = float(turn_sign)
            self._fast_turn_reached = False
            self._fast_turn_active = True

    def _disarm_fast_turn_and_stop(self):
        with self._turn_lock:
            self._fast_turn_active = False
            self.StopRobot()

    def _imu_is_fresh(self, max_age=0.25):
        return (time.time() - self._last_imu_cb_time) < max_age

    def _fast_turn_step(self, current_imu_yaw):
        with self._turn_lock:
            if not self._fast_turn_active or self._fast_turn_reached:
                return
            st = self.fsm.get_state()
            in_turn_phase = (
                st == FSMState.PATH_FOLLOWING and
                self.fsm.state_before_planning == FSMState.UTURN_PLANNING and
                getattr(self, 'uturn_step_phase', '') in ('TURN_90_TO_BLACK', 'TURN_90_INTO_ROW2')
            )
            if not in_turn_phase:
                self._fast_turn_active = False
                return

            rem_deg = self._turn_remaining_deg(current_imu_yaw)
            gyro_dps = abs(math.degrees(self.localization_manager.imu_angular_vel_z))
            # Dừng sớm một góc bằng quãng quay thêm do trễ (gyro * lead), kẹp 1.0° - 6.0°
            lead_deg = float(np.clip(gyro_dps * self.turn_stop_lead_sec, 1.0, 6.0))
            if rem_deg <= lead_deg:
                self._fast_turn_reached = True
                self._fast_turn_active = False
                self.cmd_vel_pub.publish(Twist())
                return

            phase_elapsed = self.get_clock().now().nanoseconds / 1e9 - getattr(self, 'uturn_phase_start_time', 0.0)
            twist = Twist()
            twist.angular.z = float(self._fast_turn_sign * self._turn_rate_for_remaining(rem_deg, gyro_dps, phase_elapsed))
            self.cmd_vel_pub.publish(twist)

    def imu_callback(self, msg: Imu):
        self.localization_manager.update_imu(msg)
        current_imu_yaw = self.localization_manager.imu_yaw
        self._last_imu_cb_time = time.time()

        # 0. Fast Turn Guard: cắt lệnh xoay ngay khi đủ 90° (không chờ khung hình camera)
        if self._fast_turn_active:
            self._fast_turn_step(current_imu_yaw)

        # 1. Tích lũy góc quay thực tế từ IMU khi xe đang quay đầu
        is_uturn_active = (
            self.fsm.get_state() == FSMState.UTURN_PLANNING or
            (self.fsm.get_state() == FSMState.PATH_FOLLOWING and self.fsm.state_before_planning == FSMState.UTURN_PLANNING) or
            self.is_trimming_uturn_heading
        )
        if is_uturn_active:
            if hasattr(self, 'uturn_start_imu_yaw') and self.uturn_start_imu_yaw is not None:
                dyaw_from_start = math.atan2(
                    math.sin(current_imu_yaw - self.uturn_start_imu_yaw),
                    math.cos(current_imu_yaw - self.uturn_start_imu_yaw)
                )
                self.uturn_accum_imu_yaw = abs(dyaw_from_start)
            self._prev_imu_yaw_for_turn = current_imu_yaw

            # 2. Báo cáo phát hiện trượt bánh (Slip Detection)
            if self.uturn_accum_wheel_yaw > 0.10:
                slip_deg = math.degrees(self.uturn_accum_wheel_yaw - self.uturn_accum_imu_yaw)
                if slip_deg > 8.0 and not self._slip_warning_logged:
                    self.get_logger().warn(
                        f"⚠️ [CẢNH BÁO TRƯỢT BÁNH] Bánh xe quay {math.degrees(self.uturn_accum_wheel_yaw):.1f}° "
                        f"nhưng IMU thực tế chỉ quay {math.degrees(self.uturn_accum_imu_yaw):.1f}° (Trượt {slip_deg:.1f}° trên cỏ/đất)! "
                        f"Hệ thống đang điều khiển bù góc theo IMU để đảm bảo quay đúng 180°!"
                    )
                    if self.enable_file_logging and self.telemetry_logger:
                        self.telemetry_logger.log_event(
                            "WHEEL_SLIP_DETECTED",
                            f"Bánh quay {math.degrees(self.uturn_accum_wheel_yaw):.1f}°, IMU quay {math.degrees(self.uturn_accum_imu_yaw):.1f}°, Trượt {slip_deg:.1f}°"
                        )
                    self._slip_warning_logged = True

            # 3. Tầng phân tích & nắn chỉnh cục bộ: Khép vòng IMU 50Hz + LiDAR C1 căn tâm miệng Luống 2
            if self.is_trimming_uturn_heading:
                now_sec = self.get_clock().now().nanoseconds / 1e9
                yaw_err = math.atan2(
                    math.sin(self.uturn_target_yaw - current_imu_yaw),
                    math.cos(self.uturn_target_yaw - current_imu_yaw)
                )
                trim_threshold = math.radians(3.0)  # Chuẩn sai số góc < 3.0 độ
                timed_out = (now_sec - self.uturn_trim_start_time) > 6.0
                hard_timeout = (now_sec - self.uturn_trim_start_time) > 10.0

                # Quét phân tích miệng Luống 2 bằng LiDAR C1
                docking_info = self.lidar_processor.get_row_entrance_centering(
                    min_fwd=0.20, max_fwd=1.80, expected_row_spacing=self.row_spacing
                )
                lat_offset = docking_info["lateral_offset"] if docking_info["detected"] else 0.0

                heading_ready = (abs(yaw_err) <= trim_threshold)
                centering_ready = (not docking_info["detected"]) or (abs(lat_offset) <= 0.05)

                if (heading_ready and centering_ready) or (timed_out and abs(yaw_err) <= math.radians(15.0)) or hard_timeout:
                    self.is_trimming_uturn_heading = False
                    self.StopRobot()
                    start_deg = math.degrees(self.uturn_start_imu_yaw)
                    now_deg = math.degrees(current_imu_yaw)
                    accum_deg = math.degrees(self.uturn_accum_imu_yaw)
                    status_note = "THÀNH CÔNG HOÀN HẢO" if (heading_ready and centering_ready) else "TIMEOUT AN TOÀN"
                    self.get_logger().info(
                        f"✅ [TẦNG PHÂN TÍCH CỤC BỘ - {status_note}] Đã hoàn tất quay đầu và căn chuẩn vào Luống 2! "
                        f"(Start: {start_deg:+.1f}°, Hiện tại: {now_deg:+.1f}°, IMU quay: {accum_deg:.1f}°, Lệch trục: {math.degrees(yaw_err):+.2f}°, Lệch tim LiDAR: {lat_offset:+.2f}m). "
                        f"Bàn giao quyền điều khiển cho AI CNN!"
                    )
                    if self.enable_file_logging and self.telemetry_logger:
                        self.telemetry_logger.log_event(
                            "UTURN_SUCCESS_IMU",
                            f"Quay đầu 180° thành công: Start={start_deg:.1f}°, Current={now_deg:.1f}°, Total={accum_deg:.1f}°, ErrYaw={math.degrees(yaw_err):.2f}°, LatOffset={lat_offset:.2f}m"
                        )
                    self.transition_after_path(self.get_clock().now())
                else:
                    twist = Twist()
                    if not heading_ready:
                        # Ưu tiên 1: Nắn đầu xe song song trục luống 2 với mô-men quay đầy đủ
                        twist.linear.x = 0.0
                        turn_dir = 1.0 if yaw_err > 0 else -1.0
                        twist.angular.z = turn_dir * float(self.turn_angular_speed)
                    else:
                        # Ưu tiên 2: Đầu xe đã thẳng, bò chậm nắn tim
                        twist.linear.x = 0.04
                        dock_steer = float(np.clip(lat_offset * 1.5, -0.30, 0.30))
                        twist.angular.z = dock_steer
                    self.cmd_vel_pub.publish(twist)
        else:
            self._prev_imu_yaw_for_turn = None

    def gps_callback(self, msg: NavSatFix):
        self.localization_manager.update_gps(msg)
        self.publish_gps()

    def publish_gps(self):
        gps_info = self.localization_manager.get_gps_coordinates()
        nav_msg = NavSatFix()
        nav_msg.header.stamp = self.get_clock().now().to_msg()
        nav_msg.header.frame_id = 'gps_link'
        nav_msg.latitude = gps_info['latitude']
        nav_msg.longitude = gps_info['longitude']
        nav_msg.altitude = gps_info['altitude']
        nav_msg.status.status = NavSatStatus.STATUS_FIX if gps_info['status'] == 'FIX' else NavSatStatus.STATUS_NO_FIX
        self.gps_pub.publish(nav_msg)

    # ── Odometry callback ──────────────────────────────────────────────
    def odom_callback(self, msg: Odometry):
        self.localization_manager.update_odometry(msg)
        pose = self.localization_manager.get_pose()
        self.publish_gps()

        # Check diagnostics status
        if pose["status"] == "SENSOR_FAILED":
            self.get_logger().warn(f"Localization warning: {pose['status']}.")

        x = pose["x"]
        y = pose["y"]
        self.current_yaw = pose["yaw"]
        self.current_x = x
        self.current_y = y
        self._odom_received = True

        if self._prev_odom_x is not None:
            dx = x - self._prev_odom_x
            dy = y - self._prev_odom_y
            if self.fsm.get_state() in [FSMState.TRACKING, FSMState.DRIVE_OUT]:
                self.distance_traveled += math.sqrt(dx*dx + dy*dy)

        self._prev_odom_x   = x
        self._prev_odom_y   = y

        # ── IMU/Odom Watchdog an toàn cho căn chỉnh CNN (6.0s fallback safety) ──
        if self.is_adjusting_heading and self.fsm.get_state() == FSMState.TRACKING:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            timed_out = (now_sec - self.heading_adjust_start_time) > 6.0
            
            if timed_out:
                self.is_adjusting_heading = False
                self.heading_adjust_cooldown_until = now_sec + 1.0
                self._aligned_frame_count = 0
                twist = Twist()
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                log_msg = f"⚠️ [ĐIỀU HƯỚNG CNN - WATCHDOG] Căn chỉnh quá 6.0s -> Khôi phục chạy thẳng {self.linear_speed:.2f} m/s an toàn!"
                self.get_logger().warn(log_msg)
                if self.enable_file_logging and self.telemetry_logger:
                    self.telemetry_logger.log_event("CNN_HEADING_TIMEOUT", log_msg)

        # Accumulate turn angle if rotating
        is_uturn_active = (
            self.fsm.get_state() == FSMState.UTURN_PLANNING or
            (self.fsm.get_state() == FSMState.PATH_FOLLOWING and self.fsm.state_before_planning == FSMState.UTURN_PLANNING)
        )
        if is_uturn_active:
            if self._prev_yaw_for_turn is not None:
                dyaw = self.current_yaw - self._prev_yaw_for_turn
                # Normalize dyaw to [-pi, pi]
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                self.accumulated_turn_angle += abs(dyaw)
            self._prev_yaw_for_turn = self.current_yaw
        else:
            self._prev_yaw_for_turn = None

    def _heading_adjust_timer_callback(self):
        """
        Timer 20Hz: Điều khiển vận tốc góc xoay căn chỉnh hướng êm ái, từ từ nhưng đủ lực.
        - Tăng tốc mềm (ramp-up trong 0.25s từ 0 lên turn_angular_speed ~0.35 rad/s).
        - Khi góc đã tiệm cận chuẩn (< 2.0°), giảm nhẹ tốc độ xuống 0.22 rad/s để chống văng lố.
        - An toàn: Nếu quá 6.0s chưa thẳng hàng, tự động khôi phục chạy tiếp để tránh kẹt robot.
        """
        if not self.is_adjusting_heading or self.fsm.get_state() != FSMState.TRACKING:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        elapsed = now_sec - self.heading_adjust_start_time

        # Watchdog an toàn: quá 6s tự động thoát
        if elapsed > 6.0:
            self.is_adjusting_heading = False
            self.heading_adjust_cooldown_until = now_sec + 1.0
            self._aligned_frame_count = 0
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.get_logger().warn("⚠️ [ĐIỀU HƯỚNG CNN] Quá 6s căn chỉnh -> Tự động khôi phục chạy thẳng!")
            return

        # Tăng tốc mềm với mô-men khởi động tức thì (tối thiểu 80% để bứt phá ma sát tĩnh trên đất gồ ghề)
        ramp = min(1.0, 0.80 + 0.20 * (elapsed / 0.15))
        base_w = float(getattr(self, 'turn_angular_speed', 0.28))
        
        # Điều phối tốc độ góc theo độ lệch góc thực tế (nhỏ dần khi gần thẳng hàng)
        angle_err = abs(self.smoothed_angle_deg)
        if angle_err < 0.8:
            target_w = max(0.18, 0.65 * base_w)
        elif angle_err < 1.4:
            target_w = max(0.22, 0.80 * base_w)
        else:
            target_w = base_w

        current_w = self.heading_adjust_dir * target_w * ramp

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = current_w
        self.cmd_vel_pub.publish(twist)

    # ── Image conversion ───────────────────────────────────────────────
    def convert_image(self, msg: Image) -> np.ndarray:
        if self.bridge is not None:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if msg.encoding in ('rgb8', 'bgr8'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img
        raise RuntimeError(f"Unsupported encoding: {msg.encoding}")

    def convert_depth_image(self, msg: Image) -> np.ndarray:
        if self.bridge is not None:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding in ('16UC1', 'mono16'):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        elif msg.encoding == '32FC1':
            return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")

    def depth_callback(self, msg: Image):
        """
        Nhận bản đồ chiều sâu quang học 3D từ Intel RealSense D435 (16UC1 mm hoặc 32FC1 m).
        1. Luôn lưu bản đồ chiều sâu mới nhất (self._latest_depth_raw) để YoloArrowDetector đo cự ly biển mũi tên.
        2. Phát hiện vật cản tầm thấp (mô đất, đá, rễ cây) nằm dưới mặt phẳng quét LiDAR 2D.
        """
        try:
            depth_img = self.convert_depth_image(msg)
            self._latest_depth_raw = depth_img
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._last_depth_raw_time = now_sec

            if not self.enable_depth_safety:
                return

            h, w = depth_img.shape[:2]

            # Trích xuất ROI hành lang chính diện trước mũi xe (35% - 60% chiều cao để không rọi xuống thảm cỏ sát mũi xe, 32% - 68% chiều rộng)
            ymin, ymax = int(h * 0.35), int(h * 0.60)
            xmin, xmax = int(w * 0.32), int(w * 0.68)
            roi = depth_img[ymin:ymax, xmin:xmax]

            # Chuẩn hóa về đơn vị mét
            if roi.dtype == np.uint16:
                roi_m = roi.astype(np.float32) / 1000.0
            else:
                roi_m = roi.astype(np.float32)

            # Lọc các điểm đo hợp lệ của RealSense D435 (> 0.25m đến 3.0m - loại bỏ vùng mù quang học và cản xe sát ống kính)
            valid_mask = (roi_m > 0.25) & (roi_m < 3.0) & ~np.isnan(roi_m)
            valid_pts = roi_m[valid_mask]

            if len(valid_pts) > 80:
                # Dùng phân vị thứ 5 (5th percentile) để loại bỏ nhiễu pixel đơn lẻ
                min_depth = float(np.percentile(valid_pts, 5))
                self.front_depth_dist = min_depth
                self._last_depth_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.front_depth_dist = float('inf')

            # Phanh khẩn cấp phản xạ nếu phát hiện vật cản tầm thấp sát cản trước
            if self.is_mission_started and self.fsm.get_state() == FSMState.TRACKING:
                estop_limit = min(self.depth_estop_dist, self.front_stop_dist)
                if self.front_depth_dist < estop_limit:
                    stop_twist = Twist()
                    stop_twist.linear.x = 0.0
                    stop_twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(stop_twist)
                    self.get_logger().warn(
                        f"🛑 [DEPTH D435 PHANH KHẨN CẤP] Vật cản 3D tầm thấp sát cản trước: {self.front_depth_dist:.2f}m < {estop_limit:.2f}m",
                        throttle_duration_sec=1.0
                    )
        except Exception as e:
            self.get_logger().debug(f"Depth convert failed: {e}")

    # ── Transition helper ──────────────────────────────────────────────
    def transition_to_state(self, new_state, now):
        old_state = self.fsm.get_state()
        if self.fsm.set_state(new_state):
            self.state_start_time = now
            self.is_adjusting_heading = False
            self._aligned_frame_count = 0
            if new_state == FSMState.TRACKING:
                self.tracking_controller.reset()
            elif new_state == FSMState.RECOVERY:
                self.recovery_start_x = self.current_x
                self.recovery_start_y = self.current_y
            
            # Reset inside_row flag ONLY when transitioning out of row (DRIVE_OUT / UTURN / IDLE)
            if old_state in [FSMState.TRACKING, FSMState.DRIVE_OUT] and new_state in [FSMState.UTURN_PLANNING, FSMState.IDLE]:
                self.inside_row = False
                self.high_confidence_counter = 0
                
            self.get_logger().info(f"[FSM TRANSITION] {old_state} ──► {new_state}")
            if self.telemetry_logger:
                self.telemetry_logger.log_event("FSM_TRANSITION", f"{old_state} -> {new_state}")

    # ── Remote CNN Callback ───────────────────────────────────────────
    def remote_cnn_callback(self, msg: Float32MultiArray):
        """Nhận kết quả AI tính sẵn từ Laptop (30-60 FPS, độ trễ thấp)."""
        if len(msg.data) >= 4:
            self._remote_cnn_data = {
                'heading_error': float(msg.data[0]),
                'lane_offset': float(msg.data[1]),
                'lane_center': float(msg.data[2]),
                'confidence': float(msg.data[3]),
                'latency_ms': float(msg.data[4]) if len(msg.data) > 4 else 0.0
            }
            self._remote_cnn_time = self.get_clock().now().nanoseconds / 1e9

    # ── Main image callback / FSM ──────────────────────────────────────
    def image_callback(self, msg: Image):
        # Prevent processing duplicate frames within 10ms
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        if hasattr(self, '_last_img_cb_ns') and (now_ns - self._last_img_cb_ns) < 10000000:
            return
        self._last_img_cb_ns = now_ns

        try:
            bgr_image = self.convert_image(msg)
            if getattr(self, '_waiting_camera_notified', False):
                self.get_logger().info("✅ Đã nhận tín hiệu Camera! Bắt đầu chuỗi tự hành AI.")
                self._waiting_camera_notified = False
            # Debug: Save the first image to verify camera is working (if enabled)
            if self.save_debug_imgs and not hasattr(self, '_debug_image_saved'):
                os.makedirs(self.output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(self.output_dir, 'camera_test.png'), bgr_image)
                self.get_logger().debug("--- Saved camera_test.png ---")
                self._debug_image_saved = True
        except Exception as e:
            self.get_logger().error(f"Image convert failed: {e}")
            return

        # ── Perception Processing via PerceptionManager ───────────────
        now_sec = now_ns / 1e9
        is_remote_cnn = (self._remote_cnn_data is not None) and ((now_sec - self._remote_cnn_time) < 0.6)
        external_cnn = self._remote_cnn_data if is_remote_cnn else None

        # Điều phối đồng bộ CNN (crop_row_cnn_best.onnx) & YOLO (bao1.onnx) theo trạng thái FSM:
        # - Khi đang ở pha bám Mũi tên Vàng/Đen ngoài đầu luống (DRIVE_OUT, TURN_90_TO_BLACK, APPROACH_BLACK_05M),
        #   tạm ngưng suy luận CNN nặng để dồn 100% tốc độ khung hình (FPS) cho YOLO + Depth D435 bắt mốc 0.50m!
        # - Khi ở TRACKING (bám luống 1 & 2) hoặc TURN_90_INTO_ROW2 (quay vào luống 2), chạy CNN 100% khung hình!
        cur_fsm = self.fsm.get_state()
        uturn_phase = getattr(self, 'uturn_step_phase', 'TURN_90_TO_BLACK')
        is_uturn_path = (cur_fsm == FSMState.PATH_FOLLOWING and self.fsm.state_before_planning == FSMState.UTURN_PLANNING)
        run_cnn = not (
            (cur_fsm == FSMState.DRIVE_OUT and not getattr(self, '_is_final_goal', False)) or
            (cur_fsm == FSMState.UTURN_PLANNING) or
            (is_uturn_path and uturn_phase in ['TURN_90_TO_BLACK', 'APPROACH_BLACK_05M'])
        )

        t_infer_start = time.time()
        perception = self.perception_manager.process_sensors(
            cv_image=bgr_image,
            distance_traveled=self.distance_traveled,
            rx=self.current_x,
            ry=self.current_y,
            ryaw=self.current_yaw,
            max_angle_deg=self.max_steering_angle_deg,
            inside_row=self.inside_row,
            external_cnn=external_cnn,
            run_cnn=run_cnn
        )
        self._last_inference_ms = (time.time() - t_infer_start) * 1000.0
        
        confidence = perception["confidence"]
        obstacle_detected = perception["obstacle_detected"]
        end_of_row = perception["end_of_row_detected"]
        raw_angle = perception["heading_error"]
        lane_center = perception["lane_center"]
        lane_offset = perception["lane_offset"]
        left_side_dist = perception.get("left_side_dist", float('inf'))
        right_side_dist = perception.get("right_side_dist", float('inf'))

        # Log xác nhận nhận ảnh thực tế từ Camera
        self._received_frame_count += 1
        if self._received_frame_count == 1:
            self.get_logger().info(
                f"📷 [CNN ĐÃ NHẬN ẢNH TỪ CAMERA] Frame #1: {bgr_image.shape[1]}x{bgr_image.shape[0]} | BGR8 | "
                f"Góc lái AI tính: {raw_angle:+.2f}° (Tin cậy: {confidence*100:.1f}%) | Độ trễ AI: {self._last_inference_ms:.1f}ms"
            )
        elif self._received_frame_count % 30 == 0:
            self.get_logger().info(
                f"📷 [CNN NHẬN ẢNH #{self._received_frame_count}] Góc: {raw_angle:+.2f}° | "
                f"Tin cậy: {confidence*100:.1f}% | AI: {self._last_inference_ms:.1f}ms",
                throttle_duration_sec=1.5
            )

        # ── Nhận diện Mũi tên Vàng (Điều kiện 1) và Mũi tên Đen (Điều kiện 2) ──
        # Điều phối thông minh CNN + YOLO để tối ưu FPS và không gây trễ vòng lặp điều khiển:
        # - Khi đã vào Nhịp 3 (TURN_90_INTO_ROW2) hoặc đang chạy CNN Luống 2: Tắt hoàn toàn YOLO để dồn 100% CPU cho CNN bám luống 2.
        # - Khi đang ở DRIVE_OUT hoặc Nhịp 1 & 2 (tìm Mũi tên Vàng / Đen): Chạy YOLO 100% các frame để bắt khoảng cách 0.50m tức thì.
        # - Khi đang ở đầu Luống 1: Chạy YOLO cách nhịp (mỗi 2 frame), và tự động nâng lên 100% frame ngay khi thấy Mũi tên Vàng hoặc khi hết luống (confidence < 0.35).
        arrow_res = None
        need_yolo = (
            (cur_fsm == FSMState.TRACKING and self.current_lane_idx == 1) or
            (cur_fsm == FSMState.DRIVE_OUT and not getattr(self, '_is_final_goal', False)) or
            (cur_fsm in [FSMState.UTURN_PLANNING, FSMState.PATH_FOLLOWING] and uturn_phase in ['TURN_90_TO_BLACK', 'APPROACH_BLACK_05M'])
        )
        if getattr(self, 'enable_arrow_landmarks', True) and hasattr(self, 'arrow_detector') and self.arrow_detector and need_yolo:
            prev_y_seen = bool(self._latest_arrow_res and self._latest_arrow_res['yellow_arrow'].get('raw_detected', self._latest_arrow_res['yellow_arrow']['detected']))
            run_this_frame = (cur_fsm != FSMState.TRACKING) or prev_y_seen or (confidence < self.low_confidence_threshold) or (self._received_frame_count % 2 == 0)
            if run_this_frame:
                depth_age = now_sec - getattr(self, '_last_depth_raw_time', 0.0)
                fresh_depth = getattr(self, '_latest_depth_raw', None) if depth_age < 1.0 else None
                arrow_res = self.arrow_detector.detect(bgr_image, fresh_depth)
                self._latest_arrow_res = arrow_res
            else:
                arrow_res = self._latest_arrow_res

        # Xuất bản mặt nạ nhị phân /crop_row/mask (mono8) và HUD AI thời gian thực lên /camera/ai_view
        if self.bridge is not None:
            try:
                h, w = bgr_image.shape[:2]
                mask_orig = None
                if hasattr(self, 'inference') and self.inference and getattr(self.inference, 'latest_mask', None) is not None:
                    prob_map = self.inference.latest_mask
                    mask_384 = (prob_map >= self.inference.mask_threshold).astype(np.uint8) * 255
                    mask_orig = cv2.resize(mask_384, (w, h), interpolation=cv2.INTER_NEAREST)
                    if getattr(self, 'mask_pub', None) is not None:
                        mask_msg = self.bridge.cv2_to_imgmsg(mask_orig, encoding='mono8')
                        mask_msg.header = msg.header
                        self.mask_pub.publish(mask_msg)

                if getattr(self, 'ai_view_pub', None) is not None:
                    img_cx = int((w - 1) / 2.0)
                    in_w = float(self.inference.input_size[1]) if (hasattr(self, 'inference') and self.inference) else 384.0
                    lane_cx_px = int(lane_center * (w / in_w)) if (hasattr(self, 'inference') and self.inference and run_cnn) else img_cx
                    lane_cx_px = int(np.clip(lane_cx_px, 0, w - 1))
                    hud_frame = bgr_image.copy()

                    # Phủ lớp vạch luống CNN màu xanh lá trực quan (khớp với test_onnx.py)
                    if mask_orig is not None and np.any(mask_orig > 0):
                        overlay = hud_frame.copy()
                        overlay[mask_orig > 0] = (0, 255, 0)
                        hud_frame = cv2.addWeighted(hud_frame, 0.65, overlay, 0.35, 0)

                    for y_s in range(0, h, 25):
                        cv2.line(hud_frame, (img_cx, y_s), (img_cx, min(y_s + 15, h)), (0, 255, 0), 2)
                    cv2.line(hud_frame, (lane_cx_px, h - 1), (lane_cx_px, int(h * 0.4)), (255, 140, 0), 3)

                    # Vẽ Bounding Box Mũi tên Vàng nếu phát hiện
                    if arrow_res and arrow_res['yellow_arrow']['detected']:
                        yb = arrow_res['yellow_arrow']['bbox']
                        yd = arrow_res['yellow_arrow']['depth_m']
                        cv2.rectangle(hud_frame, (yb[0], yb[1]), (yb[2], yb[3]), (0, 255, 255), 2)
                        cv2.putText(hud_frame, f"YELLOW ARROW (MOC 1): {yd:.2f}m", (yb[0], max(20, yb[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    # Vẽ Bounding Box Mũi tên Đen nếu phát hiện
                    if arrow_res and arrow_res['black_arrow']['detected']:
                        bb = arrow_res['black_arrow']['bbox']
                        bd = arrow_res['black_arrow']['depth_m']
                        bo = arrow_res['black_arrow']['offset_x']
                        cv2.rectangle(hud_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 0), 3)
                        cv2.rectangle(hud_frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 1)
                        cv2.putText(hud_frame, f"BLACK ARROW (MOC 2): {bd:.2f}m | off:{bo:+.2f}", (bb[0], max(20, bb[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    depth_text = f"Depth: {self.front_depth_dist:.2f}m" if self.front_depth_dist < 10.0 else "Depth: Clear"
                    cv2.putText(hud_frame, f"Frame #{self._received_frame_count} | Conf: {confidence*100:.1f}% | Steer: {self.smoothed_angle_deg:+.2f} deg | {depth_text}",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                    ai_msg = self.bridge.cv2_to_imgmsg(hud_frame, encoding='bgr8')
                    ai_msg.header = msg.header
                    self.ai_view_pub.publish(ai_msg)
            except Exception:
                pass

        # ── FSM ───────────────────────────────────────────────────────
        twist        = Twist()
        now          = self.get_clock().now()
        if self.node_start_time is None:
            self.node_start_time = now
            self.state_start_time = now
        elapsed_state= (now - self.state_start_time).nanoseconds / 1e9
        elapsed_total= (now - self.node_start_time).nanoseconds / 1e9
        warmup_done  = elapsed_total > self.warmup_time

        current_state = self.fsm.get_state()

        now_sec = now.nanoseconds / 1e9

        # Save diagnostic frame every 3 seconds (sim time, if enabled)
        if self.save_debug_imgs:
            if now_sec - self.last_img_save_time >= 3.0:
                self.last_img_save_time = now_sec
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    if hasattr(self.inference, 'latest_mask') and self.inference.latest_mask is not None:
                        # Resize mask to match BGR image height and width
                        mask_resized = cv2.resize(self.inference.latest_mask, (bgr_image.shape[1], bgr_image.shape[0]))
                        
                        # Convert single-channel mask (0.0 to 1.0) to binary mask (0 or 255)
                        binary_mask = (mask_resized >= self.inference.mask_threshold).astype(np.uint8) * 255
                        binary_mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                        
                        # Create overlay image: cyan highlight for segmented areas
                        overlay_image = bgr_image.copy()
                        overlay_image[binary_mask > 128] = [255, 255, 0] # Cyan BGR
                        blend_image = cv2.addWeighted(bgr_image, 0.6, overlay_image, 0.4, 0)
                        
                        # Draw guidance lines on blend_image
                        h, w = bgr_image.shape[:2]
                        image_center = (w - 1) / 2.0
                        
                        # 1. Lime dashed line for image center
                        for y_start in range(0, h, 20):
                            cv2.line(blend_image, (int(image_center), y_start), (int(image_center), min(y_start + 10, h)), (0, 255, 0), 2)
                        
                        # 2. Deepskyblue target line for lane center
                        lane_center_raw = lane_center * (w / self.inference.input_size[1])
                        lane_center_raw = np.clip(lane_center_raw, 0.0, w - 1.0)
                        line_top = int(h * 0.6)
                        cv2.line(blend_image, (int(lane_center_raw), h - 1), (int(lane_center_raw), line_top), (255, 191, 0), 3)
                        
                        # Concatenate horizontally: BGR, Mask, Overlay
                        canvas = np.hstack((bgr_image, binary_mask_colored, blend_image))
                        
                        # Add text details for easier debugging
                        cv2.putText(canvas, f"State: {current_state} | Conf: {confidence:.2f} | Dist: {self.distance_traveled:.2f}m | Steer: {self.smoothed_angle_deg:.2f} deg", 
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Save the image
                        filename = os.path.join(self.output_dir, f"frame_{int(now_sec)}.png")
                        cv2.imwrite(filename, canvas)
                        self.get_logger().debug(f"--- Saved diagnostic frame: {filename} ---")
                except Exception as save_err:
                    self.get_logger().error(f"Failed to save diagnostic frame: {save_err}")

        # ── Chế độ chờ lệnh từ RViz (Standby / Ready) ─────────────────
        if not self.is_mission_started:
            # Cập nhật các thông số góc lái và độ tin cậy từ Camera / AI
            if confidence >= self.low_confidence_threshold:
                self.has_seen_row = True
                self.smoothed_angle_deg = (
                    self.ema_alpha * raw_angle + (1.0 - self.ema_alpha) * self.smoothed_angle_deg
                )
            else:
                self.smoothed_angle_deg = raw_angle

            # Đảm bảo động cơ tuyệt đối đứng yên
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

            # Cập nhật cache trạng thái để in log 1Hz và phục vụ RViz
            self._last_image_time = time.time()
            self._latest_confidence = confidence
            self._latest_steer_deg = self.smoothed_angle_deg
            self._latest_twist = twist
            self._latest_gps_info = self.localization_manager.get_gps_coordinates()
            self._latest_inf_ms = getattr(self, '_last_inference_ms', 0.0)
            return

        # Khi Camera/CNN nhìn thấy hàng (confidence >= low_confidence_threshold):
        # Bám luống ngay lập tức dù đặt xe ở gần hay xa, thẳng hàng hay lệch xéo.
        # Không ràng buộc cứng mét chạy hay điều kiện thành tường.
        is_in_maneuver = current_state in [FSMState.UTURN_PLANNING, FSMState.AVOID_PLANNING, FSMState.PATH_FOLLOWING, FSMState.RECOVERY, FSMState.DRIVE_OUT]
        if confidence >= self.low_confidence_threshold and not is_in_maneuver:
            self.has_seen_row = True
            if not self.inside_row:
                self.inside_row = True
                if self.row_start_x is None:
                    self.row_start_x = self.current_x
                    self.current_lane_y = self.current_y
                self.row_completed = False
                self.get_logger().info(f"🌾 [BÁM HÀNG] Nhận diện luống thành công tại x={self.row_start_x:.2f}m, y={self.current_lane_y:.2f}m (conf={confidence:.2f}). Bắt đầu dẫn hướng vào tim hàng!")
                if self.enable_file_logging and self.telemetry_logger:
                    self.telemetry_logger.log_event("ENTER_ROW", f"Bám luống tại x={self.row_start_x:.2f}m, y={self.current_lane_y:.2f}m (conf={confidence:.2f})")

        # ── IDLE ──────────────────────────────────────────────────────
        if current_state == FSMState.IDLE:
            self.StopRobot()
            if confidence >= self.low_confidence_threshold:
                self.transition_to_state(FSMState.TRACKING, now)
            return

        # ── TRACKING ──────────────────────────────────────────────────
        elif current_state == FSMState.TRACKING:
            # 1. An toàn mũi xe (Bumper collision guard):
            # Kết hợp khoảng cách tối thiểu từ LiDAR 2D và Depth Map 3D của RealSense D435
            front_min_dist = perception.get("front_min_dist", float('inf'))
            now_sec = now.nanoseconds / 1e9
            depth_valid = (now_sec - getattr(self, '_last_depth_time', 0.0)) < 0.5
            if depth_valid and self.front_depth_dist < front_min_dist:
                front_min_dist = self.front_depth_dist

            if front_min_dist < self.front_stop_dist:
                self.StopRobot()
                self.get_logger().warn(
                    f"⚠️ Mũi xe sát vật cản (Mũi: {front_min_dist:.2f}m [LiDAR/Depth] < {self.front_stop_dist:.2f}m, "
                    f"Trái: {left_side_dist:.2f}m, Phải: {right_side_dist:.2f}m)! Tạm dừng chờ thoáng...",
                    throttle_duration_sec=1.0
                )
                return


            # ── 1. Tầng định tâm & Lực đẩy sườn từ LiDAR C1 (LiDAR Active Centering & Repulsion) ──
            # LiDAR C1 quét 360° đo chính xác khoảng cách mép trái và mép phải:
            # - d_right < d_left: xe lệch sang phải gần thành hàng -> cần bẻ Trái (góc âm)
            # - d_left < d_right: xe lệch sang trái gần thành hàng -> cần bẻ Phải (góc dương)
            lidar_steer_bias = 0.0
            has_lidar_corridor = (left_side_dist < 1.80 and right_side_dist < 1.80)
            
            if has_lidar_corridor:
                # Độ lệch tim vật lý đo trực tiếp từ hai hàng cây/thùng carton (m)
                # corridor_offset > 0: xe lệch sang trái; corridor_offset < 0: xe lệch sang phải
                corridor_offset = (right_side_dist - left_side_dist) / 2.0
                lidar_steer_bias = float(np.clip(corridor_offset * 18.0, -10.0, 10.0))
            elif right_side_dist < self.side_near_dist:
                # Chỉ thấy sườn phải quá sát (< side_near_dist) -> đẩy gắt sang Trái
                penetration = self.side_near_dist - right_side_dist
                lidar_steer_bias = float(np.clip(-penetration * 35.0, -12.0, -3.0))
            elif left_side_dist < self.side_near_dist:
                # Chỉ thấy sườn trái quá sát (< side_near_dist) -> đẩy gắt sang Phải
                penetration = self.side_near_dist - left_side_dist
                lidar_steer_bias = float(np.clip(penetration * 35.0, 3.0, 12.0))

            # ── 2. Hợp nhất điều khiển Đa Cảm Biến (Camera CNN + LiDAR Guard + Odometry Plan) ──
            y_det_track = bool(arrow_res and arrow_res['yellow_arrow']['detected'])
            y_dist_track = float(arrow_res['yellow_arrow']['depth_m']) if y_det_track else float('inf')
            y_off_track = float(arrow_res['yellow_arrow']['offset_x']) if y_det_track else 0.0

            if confidence >= self.low_confidence_threshold:
                # Camera AI dẫn đường chính (raw_angle đã kết hợp an toàn từ perception_manager, không cộng dồn bias 2 lần)
                fused_target_angle = raw_angle
                # Nếu đang ở cuối Luống 1 và đã thấy Mũi tên Vàng phía trước (<= 1.60m), kết hợp hướng nhẹ mũi xe về tâm Mũi tên Vàng
                if self.current_lane_idx == 1 and y_det_track and y_dist_track <= 1.60 and confidence < 0.48:
                    fused_target_angle = 0.5 * raw_angle + 0.5 * (y_off_track * 12.0)
                effective_alpha = self.ema_alpha
                self.smoothed_angle_deg = (
                    effective_alpha * fused_target_angle + (1.0 - effective_alpha) * self.smoothed_angle_deg
                )
                # Liên tục ghi nhớ góc IMU "tốt" khi CNN bám luống ổn định (conf >= 0.45 và góc lệch nhỏ):
                # Giá trị trung bình cộng dồn EMA này phản ánh đúng hướng đi thẳng của tim luống.
                if confidence >= 0.45 and abs(self.smoothed_angle_deg) < 8.0:
                    cur_imu = self.localization_manager.imu_yaw
                    if self._good_row_yaw_valid:
                        self._good_row_yaw_ema = 0.92 * self._good_row_yaw_ema + 0.08 * cur_imu
                    else:
                        self._good_row_yaw_ema = cur_imu
                        self._good_row_yaw_valid = True
            elif self.current_lane_idx == 1 and y_det_track and y_dist_track <= 2.20:
                # Khi vừa hết nhìn thấy hàng ở cuối Luống 1 và thấy Mũi tên Vàng: hướng thẳng vào tâm Mũi tên Vàng, không để cây cuối luống kéo lệch!
                self.smoothed_angle_deg = 0.50 * (y_off_track * 12.0) + 0.50 * self.smoothed_angle_deg
            elif abs(lidar_steer_bias) > 0.5 and (left_side_dist < 0.95 and right_side_dist < 0.95):
                # Chỉ dùng LiDAR giữ tim luống khi CẢ 2 BÊN SƯỜN đều còn trong hành lang hẹp (< 0.95m),
                # tránh bị 1 cây lẻ ở cuối luống đánh lái lệch hướng trước khi vào DRIVE_OUT!
                effective_alpha = max(self.ema_alpha, 0.80)
                self.smoothed_angle_deg = (
                    effective_alpha * lidar_steer_bias + (1.0 - effective_alpha) * self.smoothed_angle_deg
                )
                if now_sec - getattr(self, '_last_lidar_centering_log_time', 0.0) >= 2.0:
                    self._last_lidar_centering_log_time = now_sec
                    self.get_logger().info(
                        f"🛡️ [LIDAR ACTIVE CENTERING] Camera conf thấp ({confidence*100:.1f}%) -> LiDAR C1 tự động nắn tim luống "
                        f"(L:{left_side_dist:.2f}m, R:{right_side_dist:.2f}m -> Lái: {self.smoothed_angle_deg:+.1f}°)"
                    )
            elif self.enable_global_plan_assist and self.inside_row:
                # Fallback khi cả Camera và LiDAR đều thoáng ra bãi rộng: Dùng Odometry giữ tim luống
                target_y = 0.0 if self.current_lane_idx == 1 else self.current_lane_y
                target_yaw = 0.0 if self.current_lane_idx == 1 else (math.pi if self.current_yaw >= 0 else -math.pi)
                lat_err = self.current_y - target_y
                dir_factor = 1.0 if self.current_lane_idx == 1 else -1.0
                yaw_err = math.atan2(math.sin(target_yaw - self.current_yaw), math.cos(target_yaw - self.current_yaw))
                assist_steer_deg = float(np.clip(12.0 * lat_err * dir_factor - math.degrees(yaw_err) * 0.4, -10.0, 10.0))
                self.smoothed_angle_deg = (
                    0.25 * assist_steer_deg + 0.75 * self.smoothed_angle_deg
                )
                if now_sec - getattr(self, '_last_global_assist_log_time', 0.0) >= 2.0:
                    self._last_global_assist_log_time = now_sec
                    self.get_logger().info(
                        f"🌾 [GLOBAL PLAN ASSIST] Conf thấp ({confidence*100:.1f}%) -> Hỗ trợ giữ tim Luống {self.current_lane_idx} (y={target_y:.2f}m, dy={lat_err:.2f}m, steer={self.smoothed_angle_deg:+.1f}°)"
                    )
            else:
                self.smoothed_angle_deg *= 0.90

            # Cập nhật controller nội bộ để lưu vết sai số
            now_sec = now.nanoseconds / 1e9
            if hasattr(self, '_prev_image_time'):
                dt_actual = now_sec - self._prev_image_time
            else:
                dt_actual = 0.067
            self._prev_image_time = now_sec
            dt_actual = np.clip(dt_actual, 0.001, 1.0)
            
            _lin_smc, _ang_smc = self.StartTracking(dt_actual)

            # ── LOGIC ĐIỀU HƯỚNG CNN: UỐN LƯỢN MƯỢT MÀ LIÊN TỤC ──
            # 1. Xe luôn duy trì đà chạy tiến liên tục để vượt rãnh đất gồ ghề (không dừng tùy tiện).
            # 2. Chỉ dừng xoay tại chỗ khi góc lệch thực sự quá lớn (> 30°), sau đó trở lại chạy tiến khi <= 10°.
            stop_threshold = float(getattr(self, 'turn_in_place_threshold_deg', 30.0))
            resume_threshold = float(getattr(self, 'turn_in_place_resume_deg', 10.0))

            if self.is_adjusting_heading:
                # Kiểm tra đã thẳng hàng chưa (<= resume_threshold = 10.0°)
                if abs(self.smoothed_angle_deg) <= resume_threshold:
                    self.is_adjusting_heading = False
                    self._aligned_frame_count = 0
                    self.heading_adjust_cooldown_until = now_sec + 1.2
                    self.forward_resume_start_time = now_sec

                    # Hãm phanh êm dịu triệt tiêu quán tính quay trước khi tiến:
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)

                    self.get_logger().info(
                        f"✅ [ĐIỀU HƯỚNG CNN] ĐÃ THẲNG HÀNG "
                        f"(|góc|={abs(self.smoothed_angle_deg):.2f}° <= {resume_threshold:.2f}°)! "
                        f"Hãm xoay êm dịu và tăng tốc tiến mượt mà {self.linear_speed:.2f} m/s."
                    )
                    if self.enable_file_logging and self.telemetry_logger:
                        self.telemetry_logger.log_event(
                            "CNN_HEADING_CONFIRMED_ALIGNED",
                            f"Thẳng hàng |góc|={abs(self.smoothed_angle_deg):.2f}°. Chạy tiếp {self.linear_speed:.2f} m/s"
                        )
                else:
                    self._aligned_frame_count = 0
                    if self.smoothed_angle_deg > 0.5 and self.heading_adjust_dir > 0:
                        self.heading_adjust_dir = -1.0
                    elif self.smoothed_angle_deg < -0.5 and self.heading_adjust_dir < 0:
                        self.heading_adjust_dir = 1.0

            else:
                can_trigger = (now_sec >= getattr(self, 'heading_adjust_cooldown_until', 0.0))
                if can_trigger and abs(self.smoothed_angle_deg) > stop_threshold:
                    self.is_adjusting_heading = True
                    self.heading_adjust_start_time = now_sec
                    self.heading_adjust_dir = -1.0 if self.smoothed_angle_deg > 0 else 1.0
                    self._aligned_frame_count = 0
                    self.get_logger().info(
                        f"🌾 [ĐIỀU HƯỚNG CNN] CNN trả góc lệch cực lớn {self.smoothed_angle_deg:+.2f}° > {stop_threshold:.2f}°. "
                        f"Dừng tiến, xoay từ từ tại chỗ ({self.turn_angular_speed:.2f} rad/s) chờ thẳng hàng (<= {resume_threshold:.2f}°)..."
                    )
                    if self.enable_file_logging and self.telemetry_logger:
                        self.telemetry_logger.log_event(
                            "CNN_HEADING_STOP_ADJUST",
                            f"Lệch cực lớn {self.smoothed_angle_deg:+.2f}° > {stop_threshold:.2f}°. Dừng tiến xoay căn chỉnh."
                        )

            if self.is_adjusting_heading:
                lin_speed = 0.0
                ang_vel = 0.0  # Vận tốc góc xoay từ từ do timer 20Hz kiểm soát độc lập
            else:
                # Tăng tốc tiến mềm (0.25s ramp-up) sau khi căn chỉnh xong để chống giật/trượt bánh:
                time_since_resume = now_sec - getattr(self, 'forward_resume_start_time', 0.0)
                fwd_ramp = min(1.0, time_since_resume / 0.25) if time_since_resume < 0.25 else 1.0
                lin_speed = self.linear_speed * fwd_ramp
                
                # Bẻ lái thích ứng thông minh (Adaptive Steer & Safety Speed Control):
                # 1. Bình thường: chạy đầm êm ở đúng linear_speed (0.07 m/s), bẻ lái mượt mà
                # 2. Khi sườn xe áp sát hàng (< side_near_dist): tăng độ nhạy và dải góc lái để thoát mép hàng
                # 3. Khi rất sát nguy hiểm (< side_critical_dist / side_emergency_dist): giữ tối thiểu 0.06 m/s (85% linear_speed)
                #    để động cơ 775 220RPM luôn duy trì đủ xung & mô-men xoắn vượt cỏ/vật cản, tuyệt đối không vọt quá linear_speed!
                min_side = min(left_side_dist, right_side_dist)
                is_near_side = (min_side < self.side_near_dist)
                is_critical_side = (min_side < self.side_critical_dist)
                is_emergency_side = (min_side < self.side_emergency_dist)

                min_torque_speed = float(self.linear_speed)
                if is_emergency_side:
                    lin_speed = max(min_torque_speed, lin_speed * max(1.0, self.side_emergency_speed_ratio))
                    max_w = 0.32
                    steer_gain = 0.065
                elif is_critical_side:
                    lin_speed = max(min_torque_speed, lin_speed)
                    max_w = 0.26
                    steer_gain = 0.055
                elif is_near_side:
                    lin_speed = max(min_torque_speed, lin_speed)
                    max_w = 0.22
                    steer_gain = 0.048
                else:
                    max_w = 0.18
                    steer_gain = 0.042

                # Khâu vi phân dập tắt dao động lắc lư (Oscillation Damping):
                # Khi xe đang quay về tâm, d_steer hãm nhẹ mô-men quay, chống lắc lư hình sin qua lại giữa hai hàng
                kd_steer = 0.012
                d_steer = (self.smoothed_angle_deg - getattr(self, '_prev_steer_deg', self.smoothed_angle_deg)) / max(0.01, dt_actual)
                self._prev_steer_deg = self.smoothed_angle_deg

                ang_vel = float(np.clip(-steer_gain * self.smoothed_angle_deg - kd_steer * d_steer, -max_w, max_w))

            twist.linear.x = lin_speed
            twist.angular.z = ang_vel

            # ── Pure Perception & Mission Goals End-Of-Row Check ─────────────────────
            # 1. Điều kiện nhận biết hết hàng qua camera: 5 frame liên tiếp confidence < 30%
            if confidence < 0.30:
                self.eor_low_conf_frames += 1
            else:
                self.eor_low_conf_frames = 0
            trigger_confidence = (self.eor_low_conf_frames >= 5)

            # 2. Điều phối theo 4 Điểm Mốc Nhiệm Vụ (Mission Goals):
            if self.enable_mission_goals and warmup_done:
                if self.current_lane_idx == 1:
                    # Đang chạy Luống 1: kiểm tra đã đến cuối Luống 1 chưa (Goal 1: x = field_length)
                    dist_to_g1 = math.hypot(self.current_x - self.goal_1_x, self.current_y - self.goal_1_y)
                    reached_g1 = ((dist_to_g1 <= self.goal_tolerance) or (self.current_x >= self.goal_1_x)) and (self.distance_traveled >= self.min_row_length)

                    min_gate_x = max(float(self.min_row_length), float(self.goal_1_x) - float(self.goal_tolerance))
                    trigger_valid = (trigger_confidence or end_of_row) and (self.distance_traveled >= self.min_row_length) and (self.current_x >= min_gate_x)

                    if reached_g1 or trigger_valid:
                        self.get_logger().info(
                            f"🎯 [MỐC CUỐI LUỐNG 1] Đã đến cuối Luống 1 tại x={self.current_x:.2f}m (dist={self.distance_traveled:.2f}m)! "
                            f"Kích hoạt DRIVE_OUT chạy thẳng thoát đầu luống {self.drive_out_distance:.2f}m trước khi quay đầu..."
                        )
                        if self.enable_file_logging and self.telemetry_logger:
                            self.telemetry_logger.log_event("GOAL_1_REACHED", f"Cuối Luống 1 tại x={self.current_x:.2f}m, dist={self.distance_traveled:.2f}m")
                        self.current_lane_idx = 2
                        self.drive_out_start_x = self.current_x
                        self.drive_out_start_y = self.current_y
                        self.drive_out_start_dist = self.distance_traveled
                        self.drive_out_start_time = now_sec
                        # Dùng góc IMU trung bình "tốt" (EMA khi CNN bám ổn) thay vì góc tức thời
                        self.drive_out_target_yaw = self._good_row_yaw_ema if self._good_row_yaw_valid else self.localization_manager.imu_yaw
                        self._good_row_yaw_valid = False  # Reset sau khi đã sử dụng
                        self._drive_out_ref_valid = True  # Chốt hướng tim Luống 1 làm tham chiếu 2 lần quay 90°
                        self._is_final_goal = False
                        self.transition_to_state(FSMState.DRIVE_OUT, now)
                        return

                elif self.current_lane_idx == 2:
                    # Đang chạy Luống 2: chạy ngược chiều từ x = field_length về Goal 3 (x = 0)
                    dist_to_g3 = math.hypot(self.current_x - self.goal_3_x, self.current_y - self.goal_3_y)
                    reached_g3 = (dist_to_g3 <= self.goal_tolerance) or (self.current_x <= (self.goal_3_x + 0.15))

                    if (reached_g3 and self.distance_traveled >= self.min_row_length) or (trigger_confidence and self.distance_traveled >= self.min_row_length):
                        self.get_logger().info(
                            f"🏆 [HOÀN THÀNH NHIỆM VỤ] Đã chạy xong Luống 2 về đích Goal 3 tại x={self.current_x:.2f}m, y={self.current_y:.2f}m! "
                            f"Kích hoạt DRIVE_OUT chạy thẳng thoát hàng an toàn {self.drive_out_distance:.2f}m..."
                        )
                        if self.enable_file_logging and self.telemetry_logger:
                            self.telemetry_logger.log_event("MISSION_COMPLETE", f"Về đích Goal 3 x={self.current_x:.2f}m, y={self.current_y:.2f}m")
                        self._is_final_goal = True
                        self.drive_out_start_x = self.current_x
                        self.drive_out_start_y = self.current_y
                        self.drive_out_start_dist = self.distance_traveled
                        self.drive_out_start_time = now_sec
                        # Dùng góc IMU trung bình "tốt" nếu có
                        self.drive_out_target_yaw = self._good_row_yaw_ema if self._good_row_yaw_valid else self.localization_manager.imu_yaw
                        self._good_row_yaw_valid = False
                        self.transition_to_state(FSMState.DRIVE_OUT, now)
                        return

            elif warmup_done:
                # Ưu tiên 1: Nếu đang ở Luống 1 và YOLO đã nhìn thấy MŨI TÊN VÀNG phía trước:
                y_det = bool(arrow_res and arrow_res['yellow_arrow']['detected'])
                y_dist = float(arrow_res['yellow_arrow']['depth_m']) if y_det else float('inf')
                y_thresh = float(getattr(self, 'yellow_arrow_turn_dist', 0.50))

                if self.current_lane_idx == 1 and y_det and (0.15 < y_dist <= (y_thresh + 0.05)):
                    # Nếu xe đã tiến sát Mũi tên Vàng <= 0.50m ngay khi vừa ra khỏi Luống 1 -> Lập tức kích hoạt Quay đầu 90° lần 1!
                    self.get_logger().info(
                        f"🎯 [ĐIỀU KIỆN 1 THÀNH CÔNG TỪ LUỐNG 1] Đã chạm mốc MŨI TÊN VÀNG tại cự ly {y_dist:.2f}m <= {y_thresh:.2f}m! "
                        f"Chuyển thẳng sang chu trình quay đầu 90°+90° sang Luống 2..."
                    )
                    self.current_lane_idx = 2
                    self._is_final_goal = not self.enable_uturn
                    self.inside_row = False
                    self.StopRobot()
                    self.transition_to_state(FSMState.UTURN_PLANNING if self.enable_uturn else FSMState.IDLE, now)
                    return

                # Nếu đang ở cuối Luống 1 (confidence < 0.42 hoặc hết hàng) và YOLO đã thấy Mũi tên Vàng (<= 1.85m) -> Chuyển mượt sang DRIVE_OUT chạy thẳng tới Mũi tên Vàng
                yellow_approaching = (
                    self.current_lane_idx == 1 and y_det and (y_dist <= 1.85) and
                    (confidence < 0.42 or end_of_row)
                )

                # Thuần nhận thức cảm biến thực địa: Khi xe đã từng thấy hàng (has_seen_row) và hết nhìn thấy hàng (end_of_row / trigger_confidence) -> Chạy thẳng đoạn 1.80m (DRIVE_OUT)
                lidar_headland_open = (front_min_dist > 1.80 and left_side_dist > 0.80 and right_side_dist > 0.80)
                min_valid_dist = min(0.35, float(self.min_row_length))
                is_real_eor = yellow_approaching or (
                    getattr(self, 'has_seen_row', False) and
                    (end_of_row or trigger_confidence or (trigger_confidence and lidar_headland_open)) and
                    (self.distance_traveled >= min_valid_dist or self.eor_low_conf_frames >= 6)
                )

                if is_real_eor:
                    reason_str = f"Phát hiện MŨI TÊN VÀNG cách {y_dist:.2f}m khi hết hàng" if yellow_approaching else f"Hết nhìn thấy hàng (conf={confidence*100:.1f}% < 30%) tại x={self.current_x:.2f}m"
                    self.get_logger().info(
                        f"🌾 [HẾT HÀNG -> CHẠY THẲNG {self.drive_out_distance:.2f}m (DRIVE_OUT)] {reason_str} (quãng đường trong hàng: {self.distance_traveled:.2f}m)! "
                        f"Kích hoạt DRIVE_OUT chạy thẳng {self.drive_out_distance:.2f}m & tìm Mũi tên Vàng (<= {y_thresh:.2f}m)..."
                    )
                    if self.enable_file_logging and self.telemetry_logger:
                        self.telemetry_logger.log_event("EOR_DETECTED", f"{reason_str}, dist={self.distance_traveled:.2f}m")
                    self.drive_out_start_x = self.current_x
                    self.drive_out_start_y = self.current_y
                    self.drive_out_start_dist = self.distance_traveled
                    # Dùng góc IMU trung bình "tốt" nếu có (chính xác hơn góc tức thời khi hết nhìn thấy hàng)
                    self.drive_out_target_yaw = self._good_row_yaw_ema if self._good_row_yaw_valid else self.localization_manager.imu_yaw
                    self._good_row_yaw_valid = False
                    self._drive_out_ref_valid = True  # Chốt hướng tim Luống 1 làm tham chiếu 2 lần quay 90°
                    self.drive_out_start_time = now_sec
                    if self.current_lane_idx == 1:
                        self.current_lane_idx = 2
                        self._is_final_goal = not self.enable_uturn
                    else:
                        self._is_final_goal = True
                    self.transition_to_state(FSMState.DRIVE_OUT, now)
                    return

        # ── DRIVE_OUT ─────────────────────────────────────────────────
        elif current_state == FSMState.DRIVE_OUT:
            # Đoạn chạy thẳng 1.80m thoát đầu luống (vừa chạy thẳng giữ hướng IMU vừa dùng YOLO bắt Mũi tên Vàng <= 0.50m):
            now_sec = now.nanoseconds / 1e9
            elapsed = now_sec - getattr(self, 'drive_out_start_time', now_sec)
            
            dist_from_odom = abs(self.distance_traveled - getattr(self, 'drive_out_start_dist', self.distance_traveled))
            dist_from_xy = math.hypot(
                self.current_x - getattr(self, 'drive_out_start_x', self.current_x),
                self.current_y - getattr(self, 'drive_out_start_y', self.current_y)
            )
            # Kết hợp Odom tích lũy, khoảng cách Euclid (x,y) và quãng đường động học dự phòng (v * t) nếu odom bị trượt trên cỏ
            dist_from_kinematics = elapsed * float(self.linear_speed) * 0.85
            traveled = max(dist_from_odom, dist_from_xy, dist_from_kinematics)
            
            # An toàn cản trước trong khi thoát luống:
            front_min_dist = perception.get("front_min_dist", float('inf'))
            if front_min_dist < self.front_stop_dist:
                self.StopRobot()
                self.get_logger().warn(
                    f"⚠️ Mũi xe sát vật cản khi đang DRIVE_OUT ({front_min_dist:.2f}m < {self.front_stop_dist:.2f}m)! Tạm dừng...",
                    throttle_duration_sec=1.0
                )
                return

            target_dist = float(getattr(self, 'drive_out_distance', 1.80))
            max_time = float(getattr(self, 'drive_out_max_time', 35.0))
            y_det = bool(arrow_res and arrow_res['yellow_arrow']['detected'])
            y_dist = float(arrow_res['yellow_arrow']['depth_m']) if y_det else float('inf')
            y_off = float(arrow_res['yellow_arrow']['offset_x']) if y_det else 0.0
            y_thresh = float(getattr(self, 'yellow_arrow_turn_dist', 0.50))

            # Hoàn tất đoạn chạy thẳng DRIVE_OUT khi:
            # 1. Chạm mốc Điều kiện 1 (Mũi tên Vàng <= 0.50m), HOẶC
            # 2. Nếu CHƯA thấy Mũi tên Vàng thì chạy thẳng đủ 1.80m (target_dist);
            #    Nếu ĐANG thấy Mũi tên Vàng ở phía trước (0.50m < y_dist <= 2.20m) thì tự động nới quãng đường lên tới 2.80m để xe đi tới đúng mốc <= 0.50m mới quay!
            reached_yellow_05m = (not getattr(self, '_is_final_goal', False)) and y_det and (0.15 < y_dist <= (y_thresh + 0.03))
            effective_target_dist = (target_dist + 1.00) if (y_det and y_dist <= 2.20 and not getattr(self, '_is_final_goal', False)) else target_dist
            is_cleared = reached_yellow_05m or (traveled >= effective_target_dist) or (elapsed >= max_time)

            # KIỂM TRA ĐIỀU KIỆN 1 (MŨI TÊN VÀNG): Cự ly tới mũi tên vàng <= 0.50m
            if reached_yellow_05m:
                self.get_logger().info(
                    f"🎯 [ĐIỀU KIỆN 1 THÀNH CÔNG] Đã chạm mốc MŨI TÊN VÀNG tại cự ly {y_dist:.2f}m <= {y_thresh:.2f}m (đã tiến {traveled:.2f}m)! "
                    f"Dừng tiến và kích hoạt Quay phải {float(getattr(self, 'yellow_arrow_turn_angle_deg', 90.0)):.0f}° tìm Điều kiện 2 (Mũi tên Đen)..."
                )

            if is_cleared:
                self.get_logger().info(
                    f"✅ [DRIVE_OUT HOÀN TẤT] Đã thoát luống an toàn {traveled:.2f}m (thời gian: {elapsed:.1f}s)!"
                )
                if self.enable_file_logging and self.telemetry_logger:
                    self.telemetry_logger.log_event("DRIVE_OUT_COMPLETE", f"Thoát luống thành công {traveled:.2f}m")
                
                self.low_confidence_counter = 0
                self.smoothed_angle_deg  = 0.0
                self.distance_traveled   = 0.0
                self.inside_row          = False
                self.row_start_x         = None
                self.row_completed       = False
                self.accumulated_turn_angle = 0.0
                self.StopRobot()

                if self.enable_uturn and not getattr(self, '_is_final_goal', False):
                    self.get_logger().info("🔄 Kích hoạt quay đầu 2 nhịp 90°+90° (Mũi tên Vàng -> Mũi tên Đen -> CNN Luống 2)!")
                    self.transition_to_state(FSMState.UTURN_PLANNING, now)
                else:
                    self.get_logger().info("🛑 Chế độ thuần tracking (hoặc hoàn thành nhiệm vụ): Dừng xe an toàn ngoài bãi đất trống.")
                    self.transition_to_state(FSMState.IDLE, now)
                return

            # Khi đang tiến tới Mũi tên Vàng: giảm tốc êm dịu khi còn cách < 0.85m để dừng chính xác tại 0.50m không bị trượt quán tính
            if y_det and not getattr(self, '_is_final_goal', False):
                approach_speed = float(self.turn_linear_speed) if y_dist < 0.85 else float(self.linear_speed)
                twist.linear.x = approach_speed
                twist.angular.z = float(np.clip(-0.9 * y_off, -0.30, 0.30))
            else:
                target_yaw = getattr(self, 'drive_out_target_yaw', self.current_yaw)
                current_imu_yaw = self.localization_manager.imu_yaw
                yaw_err = math.atan2(math.sin(target_yaw - current_imu_yaw), math.cos(target_yaw - current_imu_yaw))

                # ── LiDAR corridor centering trong 0.35m đầu tiên thoát luống ──
                # Xe còn nằm giữa hàng cuối dù camera đã hết thấy cây. Nếu cả 2 bên sườn
                # đều có vật < 0.85m (tức vẫn ở giữa 2 hàng cây), dùng chênh lệch khoảng cách
                # trái-phải để giữ xe ở giữa, tránh va gốc cây khi thoát ra.
                corridor_correction = 0.0
                if traveled < 0.35 and left_side_dist < 0.85 and right_side_dist < 0.85:
                    side_diff = left_side_dist - right_side_dist  # >0 = lệch phải, <0 = lệch trái
                    corridor_correction = float(np.clip(0.8 * side_diff, -0.20, 0.20))

                twist.linear.x = self.linear_speed
                twist.angular.z = float(np.clip(1.2 * yaw_err + corridor_correction, -0.30, 0.30))
            self.cmd_vel_pub.publish(twist)
            return

        # ── REACTIVE_AVOID ────────────────────────────────────────────
        elif current_state == FSMState.REACTIVE_AVOID:
            self.StopRobot()
            
            # Check if obstacle has cleared
            if not obstacle_detected:
                self.get_logger().info("Obstacle cleared! Resuming TRACKING...")
                if self.enable_file_logging and self.telemetry_logger:
                    self.telemetry_logger.log_event("OBSTACLE_CLEAR", "Vật cản đã rời khỏi tầm quét. Quay lại TRACKING")
                self.transition_to_state(FSMState.TRACKING, now)
                return
                
            # If wait time exceeded, transition to AVOID_PLANNING (Task 9)
            if elapsed_state >= self.reactive_avoid_wait_time:
                self.get_logger().warn(f"Obstacle still present after {self.reactive_avoid_wait_time}s. Transitioning to AVOID_PLANNING...")
                self.transition_to_state(FSMState.AVOID_PLANNING, now)
                return

        # ── AVOID_PLANNING ────────────────────────────────────────────
        elif current_state == FSMState.AVOID_PLANNING:
            dir_x = 1.0 if math.cos(self.current_yaw) >= 0 else -1.0
            goal_x = self.current_x + dir_x * 4.0
            goal_y = self.current_y

            # Generate smooth lane-constrained polynomial avoidance trajectory
            waypoints = self.generate_backup_avoidance_path(goal_x, goal_y)
            if waypoints:
                self.pure_pursuit_controller.set_path(waypoints)
                self.transition_to_state(FSMState.PATH_FOLLOWING, now)
            else:
                self.get_logger().error("Avoidance path generation failed! Transitioning to RECOVERY...")
                self.transition_to_state(FSMState.RECOVERY, now)

        # ── UTURN_PLANNING ────────────────────────────────────────────
        elif current_state == FSMState.UTURN_PLANNING:
            # Xác định chiều chuyển luống và tọa độ luống đích
            target_y = -abs(self.row_spacing) if self.turn_side.upper() == 'RIGHT' else abs(self.row_spacing)
            self.current_lane_y = target_y

            # Ghi nhận mốc xuất phát quay đầu từ vị trí và cảm biến IMU
            self.uturn_start_x = self.current_x
            self.uturn_start_y = self.current_y
            self.uturn_start_imu_yaw = self.localization_manager.imu_yaw

            # Ưu tiên lấy hướng chuẩn của tim Luống 1 (drive_out_target_yaw hoặc _good_row_yaw_ema)
            # để bù lại vài độ xe đã bẻ lái khi bám Mũi tên Vàng trong DRIVE_OUT -> Quay đúng 90° vuông góc Luống 1!
            row1_yaw_candidate = self.uturn_start_imu_yaw
            if getattr(self, '_drive_out_ref_valid', False):
                row1_yaw_candidate = float(getattr(self, 'drive_out_target_yaw', self.uturn_start_imu_yaw))
                self._drive_out_ref_valid = False
            elif getattr(self, '_good_row_yaw_valid', False):
                row1_yaw_candidate = float(self._good_row_yaw_ema)
                self._good_row_yaw_valid = False

            if abs(self._wrap_angle(self.uturn_start_imu_yaw - row1_yaw_candidate)) <= math.radians(25.0):
                self.uturn_row1_yaw = row1_yaw_candidate
            else:
                self.uturn_row1_yaw = self.uturn_start_imu_yaw

            turn_sign = -1.0 if self.turn_side.upper() == 'RIGHT' else 1.0
            target_turn1_deg = float(getattr(self, 'yellow_arrow_turn_angle_deg', 90.0))

            # Hướng đích sau Nhịp 1 (vuông góc 90° so với Luống 1) & Hướng đích Luống 2 (ngược chiều 180° so với Luống 1):
            self.uturn_turn1_target_yaw = self._wrap_angle(
                self.uturn_row1_yaw + turn_sign * math.radians(target_turn1_deg)
            )
            self.uturn_target_yaw = self._wrap_angle(self.uturn_row1_yaw + math.pi)

            self.uturn_accum_imu_yaw = 0.0
            self.uturn_accum_wheel_yaw = 0.0
            self._prev_imu_yaw_for_turn = self.uturn_start_imu_yaw
            self._prev_wheel_yaw_for_turn = None
            self._slip_warning_logged = False
            self.is_trimming_uturn_heading = False

            # Reset bộ nhớ nhận diện Mũi tên Đen cho chu trình quay đầu mới:
            if hasattr(self, 'arrow_detector') and self.arrow_detector:
                self.arrow_detector.reset_confirmation()
            self._last_black_seen_time = 0.0
            self._last_black_dist = float('inf')
            self._last_black_off = 0.0
            self._last_black_seen_dist_odom = 0.0

            # Khởi tạo Chu trình Quay đầu 4 Pha:
            #   Phase 1 ('TURN_90_TO_BLACK'): Tại Mũi tên Vàng (<=0.5m), quay phải đúng 90° để nhìn thẳng vào Mũi tên Đen.
            #   Phase 2 ('APPROACH_BLACK_05M'): Chạy thẳng bám tâm Mũi tên Đen cho đến khi cách Mũi tên Đen <= 0.50m.
            #   Phase 3 ('TURN_90_INTO_ROW2'): Tại Mũi tên Đen (<=0.5m), quay phải 90° lần 2 vào miệng Luống 2.
            #   Phase 4 ('ALIGN_ROW2_CENTER'): Canh chỉnh thẳng hàng & giữa tim Luống 2 trước khi chạy CNN.
            self.uturn_step_phase = 'TURN_90_TO_BLACK'
            self.uturn_phase_start_yaw = self.uturn_row1_yaw
            self.uturn_phase_start_time = now.nanoseconds / 1e9
            self.uturn_phase_start_x = self.current_x
            self.uturn_phase_start_y = self.current_y

            # Kích hoạt Fast Turn Guard 50Hz trên luồng IMU riêng để dừng đúng 90° không bị trễ frame camera
            self._arm_fast_turn(self.uturn_turn1_target_yaw, turn_sign)

            self.get_logger().info(
                f"🔄 [QUAY ĐẦU MŨI TÊN VÀNG -> MŨI TÊN ĐEN] Bắt đầu Nhịp 1 (Quay {target_turn1_deg:.0f}° từ Mũi tên Vàng tìm Mũi tên Đen) | "
                f"IMU Luống 1={math.degrees(self.uturn_row1_yaw):+.1f}°, IMU hiện tại={math.degrees(self.uturn_start_imu_yaw):+.1f}°, "
                f"Đích Nhịp 1={math.degrees(self.uturn_turn1_target_yaw):+.1f}°..."
            )
            
            # Reset U-turn turn tracking variables
            self.uturn_start_yaw = self.current_yaw
            self.accumulated_turn_angle = 0.0
            self._prev_yaw_for_turn = self.current_yaw

            # Sinh quỹ đạo tham chiếu lên RViz
            waypoints_obj = OmegaTurnPlanner.generate_path(
                start_x=self.current_x,
                start_y=self.current_y,
                start_yaw=self.current_yaw,
                row_spacing=self.row_spacing,
                turn_side=self.turn_side,
                open_angle_deg=self.omega_open_angle_deg,
                r1=self.omega_r1,
                clearance_dist=self.omega_clearance,
                lead_in_dist=self.omega_lead_in,
            )
            waypoints = waypoints_obj.waypoints if waypoints_obj else None
            if waypoints:
                self.pure_pursuit_controller.set_path(waypoints)
            self.last_visited_lane = 'lane_lower' if target_y < 0 else 'lane_upper'
            self.transition_to_state(FSMState.PATH_FOLLOWING, now)

        # ── PATH_FOLLOWING ────────────────────────────────────────────
        elif current_state == FSMState.PATH_FOLLOWING:
            # Nếu đang trong tầng nắn chỉnh miệng luống 2 (imu_callback quản lý 50Hz), bỏ qua để tránh đè lệnh cmd_vel
            if self.is_trimming_uturn_heading:
                return

            is_uturn = (self.fsm.state_before_planning == FSMState.UTURN_PLANNING)
            
            if is_uturn:
                now_sec = now.nanoseconds / 1e9
                current_imu_yaw = self.localization_manager.imu_yaw
                gyro_dps = abs(math.degrees(getattr(self.localization_manager, 'imu_angular_vel_z', 0.0)))
                turn_sign = -1.0 if self.turn_side.upper() == 'RIGHT' else 1.0
                phase = getattr(self, 'uturn_step_phase', 'TURN_90_TO_BLACK')
                phase_elapsed = now_sec - getattr(self, 'uturn_phase_start_time', now_sec)

                # Góc quay cục bộ tính từ đầu mỗi pha
                phase_dyaw_rad = self._wrap_angle(
                    current_imu_yaw - getattr(self, 'uturn_phase_start_yaw', current_imu_yaw)
                )
                phase_turned_deg = abs(math.degrees(phase_dyaw_rad))

                # Nhận diện Mũi tên Đen (Điều kiện 2):
                # Trong pha quay đầu & tiếp cận, chấp nhận cả raw_detected (conf >= 0.32) để không bị trễ 3 frame (~0.6s)
                black_info = arrow_res['black_arrow'] if arrow_res else None
                black_raw_det = bool(black_info and (black_info.get('detected', False) or black_info.get('raw_detected', False)))
                black_conf = float(black_info.get('confidence', 0.0)) if black_info else 0.0
                black_det = black_raw_det and (black_conf >= 0.32)
                black_dist = float(black_info['depth_m']) if black_det else float('inf')
                black_off = float(black_info['offset_x']) if black_det else 0.0
                b_thresh = float(getattr(self, 'black_arrow_turn_dist', 0.50))
                target_turn1_deg = float(getattr(self, 'yellow_arrow_turn_angle_deg', 90.0))

                if black_det and math.isfinite(black_dist):
                    self._last_black_seen_time = now_sec
                    self._last_black_dist = black_dist
                    self._last_black_off = black_off
                    self._last_black_seen_yaw = current_imu_yaw

                # ── PHA 1: QUAY 90° TỪ MŨI TÊN VÀNG (ĐIỀU KIỆN 1) ĐỂ TÌM MŨI TÊN ĐEN (ĐIỀU KIỆN 2) ──
                if phase == 'TURN_90_TO_BLACK':
                    turn1_target_yaw = getattr(self, 'uturn_turn1_target_yaw', self.uturn_phase_start_yaw + turn_sign * math.radians(target_turn1_deg))
                    rem_deg = math.degrees(self._wrap_angle(turn1_target_yaw - current_imu_yaw)) * turn_sign
                    lead_deg = float(np.clip(gyro_dps * self.turn_stop_lead_sec, 1.0, 6.0))

                    # Điều kiện chốt bằng Thị giác (Visual Lock): Đã quay >= 75° VÀ thấy Mũi tên Đen nằm gần tâm khung hình camera!
                    visual_locked_black = black_det and (abs(black_off) <= 0.18) and (phase_turned_deg >= 75.0)
                    # Quay đủ 90° (Fast Turn Guard 50Hz đã cắt lệnh, hoặc góc còn lại <= lead_deg, hoặc quá thời gian an toàn 16.0s)
                    imu_90_done = (
                        self._fast_turn_reached or
                        (rem_deg <= lead_deg) or
                        (phase_turned_deg >= (target_turn1_deg - lead_deg)) or
                        (phase_elapsed >= 16.0)
                    )

                    if visual_locked_black or imu_90_done:
                        self._disarm_fast_turn_and_stop()
                        reason = (
                            f"YOLO khóa trúng MŨI TÊN ĐEN (off={black_off:+.2f}, dist={black_dist:.2f}m, góc={phase_turned_deg:.1f}°)"
                            if visual_locked_black else
                            f"Đạt góc chuẩn {target_turn1_deg:.0f}° (đã quay={phase_turned_deg:.1f}°, còn={rem_deg:+.1f}°)"
                        )
                        self.get_logger().info(
                            f"✅ [NHỊP 1 HOÀN TẤT] Đã quay phải 90° lần 1 ({reason})! "
                            f"Chuyển sang Nhịp 2: Chạy thẳng bám tâm MŨI TÊN ĐEN đến cự ly <= {b_thresh:.2f}m..."
                        )
                        self.uturn_step_phase = 'APPROACH_BLACK_05M'
                        # Giữ hướng vuông góc 90° chuẩn (hoặc hướng khóa Mũi tên Đen) làm hướng tham chiếu chạy ngang
                        self.uturn_phase_start_yaw = current_imu_yaw if visual_locked_black else turn1_target_yaw
                        self.uturn_phase_start_time = now_sec
                        self.uturn_phase_start_x = self.current_x
                        self.uturn_phase_start_y = self.current_y
                        self._last_black_seen_dist_odom = 0.0
                        return
                    else:
                        # Nếu luồng IMU 50Hz đang hoạt động tươi, để luồng IMU phát lệnh xoay & ngắt tức thì tại 90°
                        if self._imu_is_fresh() and self._fast_turn_active:
                            self._suppress_cmd_publish = True
                        else:
                            turn_rate = self._turn_rate_for_remaining(max(0.0, rem_deg), gyro_dps, phase_elapsed)
                            twist.linear.x = 0.0
                            twist.angular.z = float(turn_sign * turn_rate)

                # ── PHA 2: CHẠY TỚI MŨI TÊN ĐEN (CÁCH 0.50M) BẰNG VISUAL SERVOING ────────
                elif phase == 'APPROACH_BLACK_05M':
                    step_dist = math.hypot(
                        self.current_x - getattr(self, 'uturn_phase_start_x', self.current_x),
                        self.current_y - getattr(self, 'uturn_phase_start_y', self.current_y)
                    )
                    if black_det and math.isfinite(black_dist):
                        self._last_black_seen_dist_odom = step_dist

                    # Hỗ trợ bộ nhớ ngắn hạn (0.9s) nếu Mũi tên Đen vừa được thấy ở cự ly gần rồi khuất 1-2 frame:
                    black_recent = ((now_sec - getattr(self, '_last_black_seen_time', 0.0)) <= 0.9) and math.isfinite(getattr(self, '_last_black_dist', float('inf')))
                    dist_since_seen = max(0.0, step_dist - getattr(self, '_last_black_seen_dist_odom', step_dist))
                    eff_black_dist = black_dist if black_det else (max(0.16, self._last_black_dist - dist_since_seen) if black_recent else float('inf'))
                    eff_black_off = black_off if black_det else (self._last_black_off if black_recent else 0.0)

                    # Điều kiện dừng tiến Nhịp 2: Cự ly tới Mũi tên Đen <= 0.50m (hoặc đã đi đủ bề rộng luống row_spacing)
                    reached_black_05m = (black_det or black_recent) and (0.15 < eff_black_dist <= (b_thresh + 0.04))
                    fallback_dist_done = (step_dist >= self.row_spacing) or (phase_elapsed >= 40.0)

                    if reached_black_05m or fallback_dist_done:
                        self._disarm_fast_turn_and_stop()
                        reason = f"Cự ly MŨI TÊN ĐEN = {eff_black_dist:.2f}m <= {b_thresh:.2f}m" if reached_black_05m else f"Quãng đường ngang = {step_dist:.2f}m"
                        self.get_logger().info(
                            f"🎯 [ĐIỀU KIỆN 2 THÀNH CÔNG - NHỊP 2 HOÀN TẤT] Đã tiến sát MŨI TÊN ĐEN ({reason})! "
                            f"Kích hoạt Nhịp 3: Quay phải 90° lần 2 vào tâm Luống 2..."
                        )
                        # Tính góc đích Nhịp 3: Quay 90° từ hướng hiện tại VÀ ưu tiên khớp với hướng ngược chiều Luống 1 (+180°)
                        local_turn2_target = self._wrap_angle(current_imu_yaw + turn_sign * math.radians(90.0))
                        if abs(self._wrap_angle(self.uturn_target_yaw - local_turn2_target)) <= math.radians(22.0):
                            self.uturn_turn2_target_yaw = self.uturn_target_yaw
                        else:
                            self.uturn_turn2_target_yaw = local_turn2_target
                            self.uturn_target_yaw = local_turn2_target

                        self.uturn_step_phase = 'TURN_90_INTO_ROW2'
                        self.uturn_phase_start_yaw = current_imu_yaw
                        self.uturn_phase_start_time = now_sec
                        self._arm_fast_turn(self.uturn_turn2_target_yaw, turn_sign)
                        return
                    else:
                        # Chạy tiến về phía Mũi tên Đen ở đúng 0.07 m/s, dùng Camera YOLO giữ Mũi tên Đen ngay chính giữa khung hình
                        twist.linear.x = float(self.turn_linear_speed)
                        if black_det or black_recent:
                            # Visual Servoing: bẻ lái trực tiếp theo độ lệch tâm của Mũi tên Đen
                            twist.angular.z = float(np.clip(-1.1 * eff_black_off, -0.42, 0.42))
                            if black_det and abs(black_off) <= 0.12:
                                self.uturn_phase_start_yaw = current_imu_yaw
                        else:
                            # Giữ hướng vuông góc 90° chuẩn khi đang tiến lại gần Mũi tên Đen
                            yaw_hold_err = self._wrap_angle(self.uturn_phase_start_yaw - current_imu_yaw)
                            twist.angular.z = float(np.clip(1.2 * yaw_hold_err, -0.28, 0.28))

                # ── PHA 3: QUAY 90° LẦN 2 TỪ MŨI TÊN ĐEN VÀO MIỆNG LUỐNG 2 ──
                elif phase == 'TURN_90_INTO_ROW2':
                    turn2_target_yaw = getattr(self, 'uturn_turn2_target_yaw', self.uturn_target_yaw)
                    rem_deg = math.degrees(self._wrap_angle(turn2_target_yaw - current_imu_yaw)) * turn_sign
                    lead_deg = float(np.clip(gyro_dps * self.turn_stop_lead_sec, 1.0, 6.0))

                    # Quét miệng Luống 2 bằng LiDAR C1
                    docking_info = self.lidar_processor.get_row_entrance_centering(
                        min_fwd=0.20, max_fwd=1.80, expected_row_spacing=self.row_spacing
                    )
                    lidar_centered = docking_info["detected"] and (abs(docking_info["lateral_offset"]) <= 0.12)

                    # Bắt buộc xe phải quay đủ >= 78° (và CNN + LiDAR đã khóa thẳng tâm Luống 2) HOẶC đủ 90° theo IMU:
                    cnn_locked_row2 = (
                        (phase_turned_deg >= 78.0) and
                        (confidence >= 0.48) and
                        (abs(lane_offset) <= 0.12) and
                        (abs(raw_angle) <= 6.0)
                    )
                    lidar_cnn_ready = (phase_turned_deg >= 80.0) and lidar_centered and (confidence >= 0.40) and (abs(raw_angle) <= 8.0)
                    imu_90_second_done = (
                        self._fast_turn_reached or
                        (rem_deg <= lead_deg) or
                        (phase_turned_deg >= (90.0 - lead_deg)) or
                        (phase_elapsed >= 16.0)
                    )

                    if cnn_locked_row2 or lidar_cnn_ready or imu_90_second_done:
                        self._disarm_fast_turn_and_stop()
                        self.get_logger().info(
                            f"✅ [NHỊP 3 HOÀN TẤT - ĐÃ QUAY {phase_turned_deg:.1f}° (còn={rem_deg:+.1f}°)] "
                            f"CNN Conf={confidence*100:.1f}%, Steer={raw_angle:+.1f}°, Offset={lane_offset:+.2f} | "
                            f"Chuyển sang Nhịp 4 (ALIGN_ROW2_CENTER): Canh chỉnh thẳng hàng Luống 2 trước khi chạy CNN..."
                        )
                        self.uturn_step_phase = 'ALIGN_ROW2_CENTER'
                        self.uturn_phase_start_yaw = current_imu_yaw
                        self.uturn_phase_start_time = now_sec
                        self._row2_aligned_frames = 0
                        return
                    else:
                        if self._imu_is_fresh() and self._fast_turn_active:
                            self._suppress_cmd_publish = True
                        else:
                            turn_rate = self._turn_rate_for_remaining(max(0.0, rem_deg), gyro_dps, phase_elapsed)
                            twist.linear.x = 0.0
                            twist.angular.z = float(turn_sign * turn_rate)

                # ── PHA 4: CANH CHỈNH GIỮA LUỐNG 2 (CNN + LIDAR + IMU) TRƯỚC KHI CHẠY BÁM LUỐNG ──
                elif phase == 'ALIGN_ROW2_CENTER':
                    docking_info = self.lidar_processor.get_row_entrance_centering(
                        min_fwd=0.20, max_fwd=1.80, expected_row_spacing=self.row_spacing
                    )
                    # 1. Sai số góc IMU so với hướng chuẩn ngược chiều Luống 1 (+180°):
                    # Quy ước dấu đồng nhất với raw_angle: dương (+) = cần rẽ Phải (CW, w < 0), âm (-) = cần rẽ Trái (CCW, w > 0)
                    imu_yaw_err_rad = self._wrap_angle(self.uturn_target_yaw - current_imu_yaw)
                    imu_err_deg = -math.degrees(imu_yaw_err_rad)

                    # 2. Sai số từ CNN (raw_angle đã bao gồm LiDAR corridor centering từ perception_manager) & LiDAR Docking cửa luống:
                    docking_err_deg = -float(docking_info["lateral_offset"]) * 15.0 if docking_info["detected"] else 0.0
                    if confidence >= 0.32:
                        combined_err_deg = 0.80 * float(raw_angle) + 0.20 * imu_err_deg
                    elif docking_info["detected"]:
                        combined_err_deg = 0.60 * imu_err_deg + 0.40 * docking_err_deg
                    else:
                        combined_err_deg = imu_err_deg + (0.50 * float(raw_angle) if abs(raw_angle) > 0.5 else 0.0)

                    has_side_corridor = (left_side_dist < 0.95 and right_side_dist < 0.95)
                    is_centered_now = (
                        (confidence >= 0.35 and abs(raw_angle) <= 5.0 and abs(lane_offset) <= 0.15 and abs(imu_err_deg) <= 12.0) or
                        (abs(combined_err_deg) <= 4.0 and abs(imu_err_deg) <= 6.0 and (confidence >= 0.30 or docking_info["detected"] or has_side_corridor))
                    )

                    if is_centered_now:
                        self._row2_aligned_frames = getattr(self, '_row2_aligned_frames', 0) + 1
                    else:
                        self._row2_aligned_frames = 0

                    # Hoàn tất canh chỉnh khi đã khóa thẳng giữa luống >= 3 frame liên tiếp (hoặc sau 3.5s tối đa)
                    if self._row2_aligned_frames >= 3 or phase_elapsed >= 3.5:
                        self.get_logger().info(
                            f"🎯 [NHỊP 4 HOÀN TẤT - ĐÃ CANH THẲNG HÀNG LUỐNG 2] "
                            f"CNN Conf={confidence*100:.1f}%, Steer={raw_angle:+.1f}°, IMU_Err={imu_err_deg:+.1f}°, Offset={lane_offset:+.2f}, "
                            f"Sườn L/R={left_side_dist:.2f}/{right_side_dist:.2f}m | Bắt đầu chạy thẳng bám CNN Luống 2!"
                        )
                        if self.enable_file_logging and self.telemetry_logger:
                            self.telemetry_logger.log_event(
                                "UTURN_2STEP_SUCCESS",
                                f"Hoàn tất quay & canh giữa Luống 2: CNN_Conf={confidence:.2f}, Steer={raw_angle:.1f}°, IMU_Err={imu_err_deg:.1f}°, Offset={lane_offset:.2f}"
                            )
                        self.smoothed_angle_deg = float(raw_angle) if confidence >= 0.30 else 0.0
                        self.transition_after_path(now)
                        return
                    else:
                        # Bước 4A: Nếu góc mũi xe còn lệch > 5.5° (trong 2.2s đầu): xoay nhẹ tại chỗ chỉnh thẳng mũi xe song song Luống 2 trước
                        if abs(combined_err_deg) > 5.5 and phase_elapsed < 2.2:
                            w_align = float(np.clip(-0.055 * combined_err_deg, -0.36, 0.36))
                            if abs(w_align) < 0.28:
                                w_align = math.copysign(0.28, w_align)
                            twist.linear.x = 0.0
                            twist.angular.z = w_align
                        else:
                            # Bước 4B: Mũi xe đã thẳng (<= 5.5°) -> tiến nhẹ 0.07 m/s vừa đi vào luống vừa căn giữa 2 hàng
                            twist.linear.x = float(self.linear_speed)
                            twist.angular.z = float(np.clip(-0.065 * combined_err_deg, -0.28, 0.28))
            else:
                lin_vel, ang_vel, finished, path_idx = self.FollowPath()
                lane_center = self.current_lane_y
                returned_to_center = (abs(self.current_y - lane_center) < 0.04) and (path_idx >= 12)
                
                # Proactive chained avoidance: if already back in center corridor and sees next obstacle ahead:
                if returned_to_center and obstacle_detected:
                    self.get_logger().warn("Returned to center and detected next obstacle! Planning next avoidance...")
                    self.transition_to_state(FSMState.AVOID_PLANNING, now)
                    self.StopRobot()
                    return
                elif finished or (returned_to_center and confidence >= self.high_confidence_threshold):
                    self.get_logger().info("Avoidance maneuver completely finished! Returning to TRACKING...")
                    self.transition_after_path(now)
                else:
                    twist.linear.x = lin_vel
                    twist.angular.z = ang_vel


        # ── RECOVERY ──────────────────────────────────────────────────
        elif current_state == FSMState.RECOVERY:
            rec_twist, finished = self.Recovery()
            
            if finished:
                self.get_logger().info("Recovery maneuver successfully completed. Returning to TRACKING...")
                self.transition_to_state(FSMState.TRACKING, now)
                self.distance_traveled = 0.0
            else:
                twist = rec_twist

        # ── EMERGENCY_STOP ────────────────────────────────────────────
        elif current_state == FSMState.EMERGENCY_STOP:
            self.StopRobot()

        # Khi đang trong chu kỳ căn chỉnh góc mềm hoặc đang nắn chỉnh miệng luống, cmd_vel do callback/timer chuyên trách kiểm soát
        if not self.is_adjusting_heading and not self.is_trimming_uturn_heading and not self._suppress_cmd_publish:
            self.cmd_vel_pub.publish(twist)
        self._suppress_cmd_publish = False

        # ── Telemetry File Logging & Standardized Terminal Status Output ──
        pose_info = self.localization_manager.get_pose()
        gps_info = pose_info.get('gps', {})
        imu_info = pose_info.get('imu', {})
        inf_ms = getattr(self, '_last_inference_ms', 0.0)
        fps_val = (1000.0 / inf_ms) if inf_ms > 0 else 0.0

        if self.enable_file_logging and self.telemetry_logger:
            self.telemetry_logger.log_telemetry({
                'fsm_state': current_state,
                'x': self.current_x,
                'y': self.current_y,
                'yaw': self.current_yaw,
                'steering_angle_deg': self.smoothed_angle_deg,
                'raw_steer_deg': raw_angle,
                'lane_offset': lane_offset,
                'linear_velocity': twist.linear.x,
                'angular_velocity': twist.angular.z,
                'imu_yaw': imu_info.get('yaw', 0.0),
                'imu_angular_vel_z': imu_info.get('angular_vel_z', 0.0),
                'imu_accel_x': imu_info.get('linear_accel_x', 0.0),
                'confidence': confidence,
                'inference_ms': inf_ms,
                'fps': fps_val,
                'distance_traveled': self.distance_traveled,
                'gps_latitude': gps_info.get('latitude', 0.0),
                'gps_longitude': gps_info.get('longitude', 0.0),
                'gps_altitude': gps_info.get('altitude', 0.0),
                'gps_dms': gps_info.get('dms', ''),
                'gps_status': gps_info.get('status', 'NO_FIX')
            })

        # Update state cache for the 1Hz terminal status logger
        display_steer_deg = self.smoothed_angle_deg
        if current_state != FSMState.TRACKING and abs(twist.linear.x) > 0.01:
            display_steer_deg = math.degrees(math.atan2(twist.angular.z * 0.58, twist.linear.x))
        self._last_image_time = time.time()
        self._latest_confidence = confidence
        self._latest_steer_deg = display_steer_deg
        self._latest_twist = twist
        self._latest_gps_info = gps_info
        self._latest_inf_ms = getattr(self, '_last_inference_ms', 0.0)
        self._latest_left_side = left_side_dist
        self._latest_right_side = right_side_dist

    def status_timer_callback(self):
        """Định kỳ in trạng thái trực quan chuẩn 1 Hz ra terminal khi đang tự hành."""
        now_sec = time.time()
        current_state = self.fsm.get_state()

        # 1. Chưa nhận được frame camera nào từ lúc bật: chỉ thông báo 1 lần, KHÔNG spam mỗi giây
        if self._last_image_time == 0.0:
            if not getattr(self, '_waiting_camera_notified', False):
                self.get_logger().info("⏳ Đang chờ nhận hình ảnh từ Camera...")
                self._waiting_camera_notified = True
            return

        # 2. Bị mất tín hiệu camera giữa chừng quá 3 giây: cảnh báo 1 lần
        if (now_sec - self._last_image_time) > 3.0:
            if not getattr(self, '_camera_lost_notified', False):
                self.get_logger().warn("⚠️ Mất tín hiệu Camera quá 3 giây! Xe đang tạm dừng.")
                self._camera_lost_notified = True
            return

        self._camera_lost_notified = False

        # 3. Khi xe đang hoạt động bình thường: in 1 dòng chuẩn, súc tích, cực kỳ dễ đọc
        inf_ms = self._latest_inf_ms
        fps_val = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
        gps_status = self._latest_gps_info.get('status', 'NO_FIX')
        gps_source = self._latest_gps_info.get('source', '')
        gps_str = f" | GPS: {self._latest_gps_info.get('latitude', 0.0):.5f}°, {self._latest_gps_info.get('longitude', 0.0):.5f}°" if (gps_status == 'FIX' and gps_source == 'DIRECT_SENSOR') else ""

        l_dist = getattr(self, '_latest_left_side', float('inf'))
        r_dist = getattr(self, '_latest_right_side', float('inf'))
        lidar_str = f" | Sườn: [L:{l_dist:4.2f}m R:{r_dist:4.2f}m]" if (l_dist < 4.0 or r_dist < 4.0) else ""

        if not self.is_mission_started:
            display_state_str = "CHỜ RVIZ"
        elif self.is_adjusting_heading:
            display_state_str = "CHỈNH GÓC"
        else:
            display_state_str = str(current_state)
        status_msg = (
            f"🌾 [AI Lái Xe] Luống {self.current_lane_idx} (x={self.current_x:4.2f}m) | "
            f"Góc: {self._latest_steer_deg:+5.2f}° | "
            f"Trạng thái: [{display_state_str:^10s}] | "
            f"Tin cậy: {self._latest_confidence*100:4.1f}% | "
            f"AI: {inf_ms:2.0f}ms ({fps_val:3.1f}FPS) | "
            f"V: {self._latest_twist.linear.x:.2f}m/s"
            f"{lidar_str}"
            f"{gps_str}"
        )
        self.get_logger().info(status_msg)
        if self.enable_file_logging and self.telemetry_logger:
            self.telemetry_logger.log_event("SYSTEM_STATUS", status_msg)

    # ── Laser Scan Callback (Lá chắn an toàn phản xạ cứng 15 Hz) ─────
    def scan_callback(self, msg: LaserScan):
        self.lidar_processor.update_scan(msg)

        # PHẢN XẠ PHANH KHẨN CẤP THỜI GIAN THỰC 15Hz (< 66ms, không phụ thuộc tần số camera):
        # Nếu xe đang di chuyển mà LiDAR quét thấy nguy cơ va chạm sát cản trước (< lidar_estop_dist):
        if self.is_mission_started and self.fsm.get_state() == FSMState.TRACKING:
            f_dist = self.lidar_processor.get_min_range_in_sector(-25.0, 25.0)
            if f_dist < self.lidar_estop_dist:
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_twist)
                self.get_logger().warn(
                    f"🛑 [LIDAR LÁ CHẮN 15Hz] Phanh khẩn cấp trước mũi xe! ({f_dist:.2f}m < {self.lidar_estop_dist:.2f}m)",
                    throttle_duration_sec=1.0
                )

    def _validate_safety_thresholds(self):
        """Đảm bảo các ngưỡng sườn luôn > nửa bề rộng xe và đúng thứ tự near > critical > emergency."""
        min_margin = 0.02  # m — mép bánh phải còn cách hàng ít nhất 2cm khi kích hoạt mức tới hạn
        floor = self.robot_half_width + min_margin
        if self.side_emergency_dist < floor:
            self.get_logger().warn(
                f"side_emergency_dist={self.side_emergency_dist:.2f}m <= nửa bề rộng xe "
                f"({self.robot_half_width:.2f}m) -> bánh đã chạm hàng trước khi kích hoạt! Nâng lên {floor:.2f}m."
            )
            self.side_emergency_dist = floor
        if self.side_critical_dist < self.side_emergency_dist:
            self.get_logger().warn("side_critical_dist < side_emergency_dist -> đặt bằng side_emergency_dist.")
            self.side_critical_dist = self.side_emergency_dist
        if self.side_near_dist < self.side_critical_dist:
            self.get_logger().warn("side_near_dist < side_critical_dist -> đặt bằng side_critical_dist.")
            self.side_near_dist = self.side_critical_dist
        if self.lidar_estop_dist > self.front_stop_dist:
            self.get_logger().warn("lidar_estop_dist > front_stop_dist -> đặt bằng front_stop_dist.")
            self.lidar_estop_dist = self.front_stop_dist
        self.get_logger().info(
            f"[SAFETY] half_width={self.robot_half_width:.2f} | front_stop={self.front_stop_dist:.2f} | "
            f"estop={self.lidar_estop_dist:.2f} | side near/crit/emerg="
            f"{self.side_near_dist:.2f}/{self.side_critical_dist:.2f}/{self.side_emergency_dist:.2f}m"
        )


    # ── Safe Row Avoidance path generator ───────────────────────────
    def generate_backup_avoidance_path(self, goal_x, goal_y):
        self.get_logger().warn("Generating smooth Quintic Polynomial avoidance trajectory with wide safety buffer...")
        rx = self.current_x
        ry = self.current_y
        ryaw = self.current_yaw
        dir_x = 1.0 if math.cos(ryaw) >= 0 else -1.0
        
        # Extract full geometric obstacle info from LiDAR in global frame
        obs_info = self.lidar_processor.get_front_obstacle_info(rx, ry, ryaw, max_dist=1.80)
        lane_center = obs_info["lane_center"]
        x_obs = obs_info["x_obs"]
        side = obs_info["side"]
        
        # Target lateral shift: Tối đa 0.04m (trong luống hẹp 1.0m với xe rộng 0.58m, không được lệch quá 4cm)
        shift_amount = 0.04
        if side == "LEFT":
            nudge_y = lane_center - (shift_amount if dir_x > 0 else -shift_amount)
            self.get_logger().info(f"Obstacle on LEFT at x={x_obs:.2f}m, y={obs_info['y_obs']:.2f}m -> Gentle weave to y={nudge_y:.2f}m")
        else:
            nudge_y = lane_center + (shift_amount if dir_x > 0 else -shift_amount)
            self.get_logger().info(f"Obstacle on RIGHT at x={x_obs:.2f}m, y={obs_info['y_obs']:.2f}m -> Gentle weave to y={nudge_y:.2f}m")

        # Hard clamp nudge_y trong biên an toàn cực hẹp
        nudge_y = float(np.clip(nudge_y, lane_center - 0.05, lane_center + 0.05))

        waypoints = []
        
        # ── Stage 1: Weave Out (kéo dài tối thiểu 0.85m để góc lái êm ái < 10 độ, không bẻ giật 53 độ)
        x_weave_end = x_obs - dir_x * 0.35
        weave_length = max(0.85, dir_x * (x_weave_end - rx))
        for t in np.linspace(0.05, 1.0, 10):
            s = 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)
            wp_x = rx + dir_x * (t * weave_length)
            wp_y = ry + s * (nudge_y - ry)
            waypoints.append([wp_x, wp_y])
            
        # ── Stage 2: Parallel Clearance Corridor (Past x_obs by 0.30m)
        x_clear_end = x_obs + dir_x * 0.30
        clear_length = max(0.40, dir_x * (x_clear_end - (rx + dir_x * weave_length)))
        for d in np.linspace(0.06, clear_length, 6):
            wp_x = rx + dir_x * (weave_length + d)
            wp_y = nudge_y
            waypoints.append([wp_x, wp_y])
            
        # ── Stage 3: Smooth Quintic Polynomial Return (kéo dài 0.70m trở về giữa luống)
        x_return_start = rx + dir_x * (weave_length + clear_length)
        return_length = 0.70
        for t in np.linspace(0.05, 1.0, 8):
            s = 10.0 * (t**3) - 15.0 * (t**4) + 6.0 * (t**5)
            wp_x = x_return_start + dir_x * (t * return_length)
            wp_y = nudge_y + s * (lane_center - nudge_y)
            waypoints.append([wp_x, wp_y])
            
        return Path(waypoints, planner_type="SafeCropAvoidance")

    # ── Transition after path completed ─────────────────────────────
    def transition_after_path(self, now=None):
        if now is None:
            now = self.get_clock().now()
            
        is_uturn = (self.fsm.state_before_planning == FSMState.UTURN_PLANNING)
        preseeded_angle = float(getattr(self, 'smoothed_angle_deg', 0.0))
        
        self.get_logger().info("Path following completed. Returning to TRACKING...")
        self.last_path_completion_time = now.nanoseconds / 1e9
        self.transition_to_state(FSMState.TRACKING, now)
        self.low_confidence_counter = 0
        self.smoothed_angle_deg = preseeded_angle if is_uturn else 0.0
        self.distance_traveled = 0.0
        
        if is_uturn:
            self.inside_row = True  # Arm inside_row immediately so obstacle avoidance is active in Row 2!
            self.row_start_x = self.current_x
            self.row_completed = False
            self.eor_detected = False
            self.accumulated_turn_angle = 0.0
            self.eor_low_conf_frames = 0
            now_sec = now.nanoseconds / 1e9
            self.forward_resume_start_time = now_sec
            self.heading_adjust_cooldown_until = now_sec + 2.0
            self.is_adjusting_heading = False
            self._aligned_frame_count = 0

            # ── Chấm điểm & Đánh giá hiệu năng quay đầu Omega Turn ─────
            if hasattr(self, 'omega_evaluator') and self.omega_evaluator.is_evaluating:
                eval_res = self.omega_evaluator.finalize_evaluation(
                    final_confidence=getattr(self, '_latest_confidence', 0.0),
                    final_x=self.current_x,
                    final_y=self.current_y,
                    final_yaw=self.current_yaw
                )
                summary_text = self.omega_evaluator.get_summary_text()
                self.get_logger().info(f"\n{summary_text}")
                if self.enable_file_logging and self.telemetry_logger:
                    self.telemetry_logger.log_event("OMEGA_EVALUATION", summary_text)
                    if hasattr(self.telemetry_logger, 'log_turn_evaluation'):
                        self.telemetry_logger.log_turn_evaluation(eval_res)
        else:
            self.inside_row = True

    def destroy_node(self):
        """Clean shutdown hook to stop robot and record final session metrics."""
        try:
            self.StopRobot()
        except Exception:
            pass
        if getattr(self, 'enable_file_logging', False) and getattr(self, 'telemetry_logger', None):
            try:
                run_dur = 0.0
                if self.node_start_time is not None:
                    run_dur = (self.get_clock().now() - self.node_start_time).nanoseconds / 1e9
                summary_msg = (
                    f"cnn_driver node stopped cleanly. "
                    f"Total duration: {run_dur:.1f}s | "
                    f"Total distance: {self.distance_traveled:.2f}m | "
                    f"Final FSM: {self.fsm.get_state()}"
                )
                self.telemetry_logger.log_event("SYSTEM_STOP", summary_msg)
                if hasattr(self.telemetry_logger, 'finalize_run'):
                    self.telemetry_logger.finalize_run()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CnnDriverNode()
    # 2 luồng: luồng 1 cho camera/odom/scan (giữ nguyên tuần tự như cũ), luồng 2 riêng cho IMU (Fast Turn Guard)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
