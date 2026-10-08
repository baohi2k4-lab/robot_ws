import math

class LidarProcessor:
    """
    Decoupled Lidar processing and obstacle detection layer.
    """
    def __init__(self, effective_max_dist=0.45, corridor_lat=0.12):
        self.latest_scan = None
        self.effective_max_dist = effective_max_dist
        self.corridor_lat = corridor_lat

    def update_scan(self, msg):
        self.latest_scan = msg

    def check_obstacle_in_front(self, rx=0.0, ry=0.5, ryaw=0.0, max_dist=1.50, inside_row=True):
        if self.latest_scan is None:
            return False

        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        # Với luống bắp thực tế: hàng cách hàng 0.9m (tim xe cách mép hàng 0.45m), cây cách cây 0.8m.
        # Xe rộng 0.58m. 
        # Để tránh nhận nhầm cây phía trước (cách 0.8m) hoặc mép thùng ở 2 bên sườn (|y| >= 0.30m):
        # - effective_max_dist = 0.45m (nhỏ hơn khoảng cách cây 0.8m, xe chạy 0.1m/s phanh an toàn trong < 5cm)
        # - corridor_lat = 0.12m (hành lang trực diện tim xe, loại trừ hoàn toàn mép thùng/hàng 2 bên)
        effective_max_dist = self.effective_max_dist
        corridor_lat = self.corridor_lat

        obstacle_hits = 0
        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.12 or r > effective_max_dist:
                continue

            angle = angle_min + idx * angle_increment
            x_fwd = r * math.cos(angle)
            y_lat = r * math.sin(angle)
            
            # Kiểm tra vật cản nằm trực diện trong hành lang tim xe
            if 0.12 < x_fwd <= effective_max_dist and abs(y_lat) <= corridor_lat:
                obstacle_hits += 1
                if obstacle_hits >= 3:  # Cần ít nhất 3 tia LiDAR liên tiếp để loại bỏ nhiễu bụi/ngọn cỏ
                    return True
        return False

    def get_obstacles_global(self, rx, ry, ryaw, max_dist=6.0):
        if self.latest_scan is None:
            return []

        obstacles = []
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.15 or r > max_dist:
                continue

            angle = angle_min + idx * angle_increment
            
            # Local coordinates
            ox_local = r * math.cos(angle)
            oy_local = r * math.sin(angle)
            
            # Global coordinates
            ox_global = rx + ox_local * math.cos(ryaw) - oy_local * math.sin(ryaw)
            oy_global = ry + ox_local * math.sin(ryaw) + oy_local * math.cos(ryaw)
            
            obstacles.append([ox_global, oy_global])

        return obstacles

    def get_min_range_in_sector(self, min_angle_deg=-25.0, max_angle_deg=25.0):
        if self.latest_scan is None:
            return float('inf')

        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        min_rad = math.radians(min_angle_deg)
        max_rad = math.radians(max_angle_deg)

        min_dist = float('inf')
        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.08:
                continue
            angle = angle_min + idx * angle_increment
            if min_rad <= angle <= max_rad:
                if r < min_dist:
                    min_dist = r
        return min_dist

    def get_obstacle_lateral_offset(self, max_dist=1.60, robot_half_width=0.35):
        """
        Returns average lateral y position (in robot frame) of obstacle in front corridor.
        Positive y = obstacle is on LEFT.
        Negative y = obstacle is on RIGHT.
        """
        if self.latest_scan is None:
            return 0.0
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges
        
        y_pts = []
        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.08 or r > max_dist:
                continue
            angle = angle_min + idx * angle_increment
            x_fwd = r * math.cos(angle)
            y_lat = r * math.sin(angle)
            if 0.08 < x_fwd < max_dist and abs(y_lat) <= robot_half_width:
                y_pts.append(y_lat)
        if len(y_pts) > 0:
            return float(sum(y_pts) / len(y_pts))
        return 0.0

    def get_front_obstacle_info(self, rx=0.0, ry=0.5, ryaw=0.0, max_dist=1.80):
        """
        Extracts comprehensive obstacle geometric properties in global frame (longitudinal distance,
        lateral position, span, and optimal side to bypass).
        """
        if self.latest_scan is None:
            return {"detected": False, "dist": 1.0, "x_obs": rx + 1.0, "y_obs": ry, "lane_center": ry, "side": "LEFT"}

        # Dùng vị trí ry thực tế của robot làm trục quy chiếu thay vì giả định cứng 0.5m
        lane_center = ry
        dir_x = 1.0 if math.cos(ryaw) >= 0 else -1.0
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        obs_points = []
        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.12 or r > max_dist:
                continue

            beam_yaw = ryaw + (angle_min + idx * angle_increment)
            xg = rx + r * math.cos(beam_yaw)
            yg = ry + r * math.sin(beam_yaw)
            
            fwd_dist = dir_x * (xg - rx)
            lat_err = abs(yg - lane_center)
            
            # Chỉ coi là chướng ngại vật phía trước nếu nằm trong khoảng hẹp (|lat_err| <= 0.14m)
            if 0.12 < fwd_dist <= max_dist and lat_err <= 0.14:
                obs_points.append((fwd_dist, xg, yg))

        if not obs_points:
            return {"detected": False, "dist": 1.0, "x_obs": rx + 1.0, "y_obs": lane_center, "lane_center": lane_center, "side": "LEFT"}

        # Sort by distance
        obs_points.sort(key=lambda p: p[0])
        min_dist = obs_points[0][0]
        x_obs = obs_points[0][1]
        
        # Average Y of front obstacle cluster
        front_cluster = [p for p in obs_points if p[0] <= min_dist + 0.25]
        y_obs = float(sum(p[2] for p in front_cluster) / len(front_cluster))
        
        # When moving in forward row (+X): yg > lane_center means obstacle on LEFT
        # When moving in return row (-X): yg > lane_center means obstacle on RIGHT
        is_fwd = (dir_x > 0)
        if is_fwd:
            side = "LEFT" if (y_obs >= lane_center) else "RIGHT"
        else:
            side = "RIGHT" if (y_obs >= lane_center) else "LEFT"

        return {
            "detected": True,
            "dist": min_dist,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "lane_center": lane_center,
            "side": side
        }

    def get_row_entrance_centering(self, min_fwd=0.20, max_fwd=1.80, expected_row_spacing=1.20):
        """
        Phân tích miệng Luống 2 bằng LiDAR C1 khi xe quay đầu hướng vào luống mới.
        Quét cụm thân cây bên sườn trái (y > 0) và sườn phải (y < 0).
        Trả về:
          - detected: bool (nhận dạng thấy cả 2 mép luống)
          - lateral_offset: float (độ lệch tim xe so với tim luống: dương = xe lệch phải cần sang trái, âm = lệch trái)
          - left_dist: float (khoảng cách mép trái)
          - right_dist: float (khoảng cách mép phải)
          - measured_spacing: float (khoảng cách đo được giữa 2 hàng luống)
        """
        if self.latest_scan is None:
            return {"detected": False, "lateral_offset": 0.0, "left_dist": 0.6, "right_dist": 0.6, "measured_spacing": expected_row_spacing}

        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        left_points_y = []
        right_points_y = []

        half_spacing = expected_row_spacing / 2.0
        max_lateral = half_spacing + 0.35

        for idx, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < 0.10 or r > 3.0:
                continue

            angle = angle_min + idx * angle_increment
            x_fwd = r * math.cos(angle)
            y_lat = r * math.sin(angle)

            # Xét các điểm phía trước mũi xe trong hành lang tiếp cận luống
            if min_fwd <= x_fwd <= max_fwd:
                if 0.15 <= y_lat <= max_lateral:
                    left_points_y.append(y_lat)
                elif -max_lateral <= y_lat <= -0.15:
                    right_points_y.append(abs(y_lat))

        if len(left_points_y) >= 3 and len(right_points_y) >= 3:
            left_points_y.sort()
            right_points_y.sort()
            # Lấy phân vị thấp (điểm mép luống gần tim xe nhất)
            d_left = left_points_y[len(left_points_y) // 4]
            d_right = right_points_y[len(right_points_y) // 4]

            # Độ lệch tim xe (dương: tim xe lệch phải so với tim luống -> cần rẽ trái nắn tim)
            lateral_offset = (d_left - d_right) / 2.0
            measured_spacing = d_left + d_right

            return {
                "detected": True,
                "lateral_offset": float(lateral_offset),
                "left_dist": float(d_left),
                "right_dist": float(d_right),
                "measured_spacing": float(measured_spacing)
            }

        return {"detected": False, "lateral_offset": 0.0, "left_dist": 0.6, "right_dist": 0.6, "measured_spacing": expected_row_spacing}


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan

    rclpy.init(args=args)
    processor = LidarProcessor()

    class LidarDiagnosticNode(Node):
        def __init__(self):
            super().__init__('lidar_processor_node')
            self.sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
            self.get_logger().info('LidarProcessor diagnostic node started, listening to /scan...')

        def scan_cb(self, msg):
            processor.update_scan(msg)
            obs = processor.check_obstacle_in_front()
            if obs:
                self.get_logger().warn('⚠️ Obstacle detected in front driving corridor!')

    node = LidarDiagnosticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
