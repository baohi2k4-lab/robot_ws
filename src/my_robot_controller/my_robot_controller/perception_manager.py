import time
import numpy as np

class SensorPriorityManager:
    """
    Multi-sensor dynamic priority selector and fault-tolerant voting system.
    As long as >= 1 sensor is operational, the system tracks smoothly.
    State transitions require consensus across active sensor modalities.
    """
    def __init__(self, cnn_threshold=0.30):
        self.cnn_threshold = cnn_threshold
        self.active_sensor = "CAMERA_CNN"

    def select_active_tracking_sensor(self, cnn_confidence, lidar_available=True):
        if cnn_confidence >= self.cnn_threshold:
            self.active_sensor = "CAMERA_CNN"
        elif lidar_available:
            self.active_sensor = "LIDAR_GEOMETRY"
        else:
            self.active_sensor = "GPS_ODOMETRY"
        return self.active_sensor

    def evaluate_eor_consensus(self, camera_eor, lidar_eor, odom_eor=False):
        """
        Fault-tolerant Voting: End of Row is confirmed when Primary sensor reports EOR
        and at least 1 secondary sensor concurs.
        """
        votes = [camera_eor, lidar_eor, odom_eor]
        return votes.count(True) >= 2


class EndOfRowDetector:
    """
    Combines multi-modal perception (Camera CNN confidence + LiDAR sector clearance)
    to detect end of corn row dynamically without false triggers inside stalk gaps.
    """
    def __init__(self, min_row_distance=2.0, low_confidence_threshold=0.30, consecutive_frames=5):
        self.min_row_distance = min_row_distance
        self.low_confidence_threshold = low_confidence_threshold
        self.consecutive_frames = consecutive_frames
        self.low_confidence_counter = 0
        self.lidar_clearance_counter = 0

    def detect(self, confidence, distance_traveled=0.0,
               left_side_dist=float('inf'), right_side_dist=float('inf'), 
               rear_left_dist=float('inf'), rear_right_dist=float('inf'), 
               front_min_dist=float('inf'), inside_row=True):
        if not inside_row:
            self.low_confidence_counter = 0
            self.lidar_clearance_counter = 0
            return False

        # 1. Vision check: Camera mất dấu luống bắp liên tục 5 frame (thuần nhận thức, không set cứng cự ly)
        if confidence < self.low_confidence_threshold:
            self.low_confidence_counter += 1
        else:
            self.low_confidence_counter = 0
        
        camera_eor = (self.low_confidence_counter >= self.consecutive_frames)
        
        # 2. LiDAR check: Chỉ kích hoạt khi phía trước trống (> 2.0m), hai bên sườn trống (> 0.85m)
        # VÀ Camera cũng mất dấu luống (confidence < low_confidence_threshold).
        if front_min_dist > 2.00 and left_side_dist > 0.85 and right_side_dist > 0.85 and (confidence < self.low_confidence_threshold):
            self.lidar_clearance_counter += 1
        else:
            self.lidar_clearance_counter = 0

        lidar_eor = (self.lidar_clearance_counter >= self.consecutive_frames)
        
        if camera_eor or lidar_eor:
            return True
        return False


class PerceptionManager:
    """
    Manages and fuses feeds from Camera/CNN and Lidar using SensorPriorityManager.
    """
    def __init__(self, inference_handler=None, lidar_processor=None, eor_detector=None,
                 front_stop_dist=0.22, side_danger_dist=0.42):
        self.inference = inference_handler
        self.lidar = lidar_processor
        self.eor_detector = eor_detector if eor_detector is not None else EndOfRowDetector()
        self.sensor_priority = SensorPriorityManager()
        self.front_stop_dist = float(front_stop_dist)    # m — vật cản trước mũi -> obstacle_detected
        self.side_danger_dist = float(side_danger_dist)  # m — ngưỡng kích hoạt Virtual Repulsion sườn

    def process_sensors(self, cv_image, distance_traveled, rx=0.0, ry=0.0, ryaw=0.0, max_angle_deg=3.5, inside_row=True, external_cnn=None):
        """
        Runs inference and checks lidar, returning standardized PerceptionOutput.
        """
        timestamp = time.time()
        
        # 1. Camera / CNN Processing
        heading_error = 0.0
        lane_offset = 0.0
        lane_center = 0.0
        confidence = 0.0
        
        if external_cnn is not None:
            # Nhận trực tiếp kết quả tính toán AI từ Laptop (Offloading - giải phóng CPU Pi)
            heading_error = external_cnn.get('heading_error', 0.0)
            lane_offset = external_cnn.get('lane_offset', 0.0)
            lane_center = external_cnn.get('lane_center', 0.0)
            confidence = external_cnn.get('confidence', 0.0)
        elif self.inference is not None and cv_image is not None:
            heading_error, lane_offset, lane_center, confidence = self.inference.process_image(cv_image, max_angle_deg)

        # 2. Lidar Processing
        obstacle_detected = False
        front_min_dist = float('inf')
        left_side_dist = float('inf')
        right_side_dist = float('inf')
        rear_left_dist = float('inf')
        rear_right_dist = float('inf')
        obstacles = []
        
        if self.lidar is not None:
            # 1. Đo khoảng cách trực diện mũi xe (-25° đến 25°)
            front_min_dist = self.lidar.get_min_range_in_sector(-25.0, 25.0)
            
            # 2. Quét bao quát toàn bộ sườn trái (20° đến 110°) từ cản trước đến bánh sau
            left_side_dist = min(
                self.lidar.get_min_range_in_sector(20.0, 70.0),
                self.lidar.get_min_range_in_sector(70.0, 110.0)
            )
            # 3. Quét bao quát toàn bộ sườn phải (-110° đến -20°) từ cản trước đến bánh sau
            right_side_dist = min(
                self.lidar.get_min_range_in_sector(-70.0, -20.0),
                self.lidar.get_min_range_in_sector(-110.0, -70.0)
            )
            rear_left_dist = self.lidar.get_min_range_in_sector(70.0, 135.0)
            rear_right_dist = self.lidar.get_min_range_in_sector(-135.0, -70.0)
            obstacles = self.lidar.get_obstacles_global(rx, ry, ryaw)
            
            # Cảnh báo phanh khẩn cấp nếu có vật cản cứng ngay trước cản xe (< front_stop_dist)
            # (Khoảng cách sườn được xử lý bằng tầng né sườn khẩn cấp Virtual Repulsion & giảm tốc, tránh dừng cứng gây kẹt xe)
            obstacle_detected = (front_min_dist < self.front_stop_dist)
            if not obstacle_detected and confidence < 0.30:
                obstacle_detected = self.lidar.check_obstacle_in_front(rx, ry, ryaw, inside_row=inside_row)

        # 3. ĐIỀU HƯỚNG DUNG HỢP (CAMERA CNN DẪN ĐƯỜNG XA + LIDAR LÁ CHẮN AN TOÀN VẬT LÝ):
        # Bán bề rộng xe đến mép bánh ngoài = 0.34m. Lối luống rộng 1.20m (tâm xe cách mép 0.60m).
        if self.lidar is not None:
            # 3.1. HỖ TRỢ ĐỊNH TÂM LIÊN TỤC BẰNG LIDAR (LiDAR Active Centering Assist)
            # Khi cả 2 bên thành hàng đều nằm trong tầm quét hợp lý (0.20m - 1.10m):
            if (0.20 < left_side_dist < 1.10) and (0.20 < right_side_dist < 1.10):
                corridor_width = left_side_dist + right_side_dist
                if 0.70 <= corridor_width <= 1.60:
                    delta_side = right_side_dist - left_side_dist  # >0: gần hàng trái hơn -> cần bẻ Phải
                    # Mỗi 10cm lệch sườn tạo lực bẻ lái ~3.5° về tim luống
                    lidar_centering_bias = float(np.clip((delta_side / 0.20) * 7.0, -8.0, 8.0))
                    
                    if confidence >= 0.30:
                        # Dung hợp cùng CNN: giữ định hướng dài hạn của CNN kết hợp kéo tâm vật lý của LiDAR
                        heading_error = float(np.clip(heading_error + lidar_centering_bias, -max_angle_deg, max_angle_deg))
                    else:
                        # Khi camera mất dấu / mờ: LiDAR đảm nhận toàn quyền định tâm
                        heading_error = lidar_centering_bias

            # 3.2. VÙNG ĐỆM AN TOÀN SƯỜN KHẨN CẤP (Emergency Virtual Repulsion - BẤT KỂ CAMERA)
            # Kích hoạt BẤT KỂ CAMERA ĐANG BÁO GÌ khi sườn xe tiến quá sát hàng (< side_danger_dist, mặc định 0.42m: mép bánh cách hàng < 8cm)
            danger_threshold = self.side_danger_dist

            if left_side_dist < danger_threshold:
                penetration = danger_threshold - left_side_dist
                # Lực đẩy mạnh mẽ tỷ lệ theo độ lấn ranh giới (ít nhất 4.0°, tối đa max_angle_deg)
                repulsion_deg = float(np.clip((penetration / 0.08) * max_angle_deg + 4.0, 4.0, max_angle_deg))
                # Bắt buộc bẻ lái sang Phải (dương) để cứu sườn trái
                heading_error = min(max(heading_error + repulsion_deg, repulsion_deg), max_angle_deg)
                
            elif right_side_dist < danger_threshold:
                penetration = danger_threshold - right_side_dist
                # Lực đẩy mạnh mẽ tỷ lệ theo độ lấn ranh giới (âm để bẻ Trái)
                repulsion_deg = float(np.clip((penetration / 0.08) * max_angle_deg + 4.0, 4.0, max_angle_deg))
                # Bắt buộc bẻ lái sang Trái (âm) để cứu sườn phải
                heading_error = max(min(heading_error - repulsion_deg, -repulsion_deg), -max_angle_deg)

        # 4. Xác định cảm biến đang chiếm quyền điều khiển
        active_sensor = self.sensor_priority.select_active_tracking_sensor(
            confidence, lidar_available=(self.lidar is not None)
        )

        end_of_row = self.eor_detector.detect(
            confidence=confidence,
            distance_traveled=distance_traveled,
            left_side_dist=left_side_dist,
            right_side_dist=right_side_dist,
            rear_left_dist=rear_left_dist,
            rear_right_dist=rear_right_dist,
            front_min_dist=front_min_dist,
            inside_row=inside_row
        )

        return {
            "heading_error": heading_error,
            "lane_offset": lane_offset,
            "lane_center": lane_center,
            "lane_width": 1.00,  # standardized/expected corn row width
            "confidence": confidence,
            "active_sensor": active_sensor,
            "obstacle_detected": obstacle_detected,
            "front_min_dist": front_min_dist,
            "left_side_dist": left_side_dist,
            "right_side_dist": right_side_dist,
            "obstacles": obstacles,
            "end_of_row_detected": end_of_row,
            "timestamp": timestamp
        }
