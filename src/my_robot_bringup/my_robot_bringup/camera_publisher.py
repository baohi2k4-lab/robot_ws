#!/usr/bin/env python3
"""
Camera Publisher Node — Hỗ trợ Intel RealSense D435 (Dual RGB + Depth Map) và iPhone / USB Webcam / RTSP Stream.

Tính năng:
  - Tự động nhận diện (video_device='auto'):
      1. Ưu tiên 1: RealSense D435 nếu đang cắm:
         - Phát đồng thời 2 luồng song song:
           + Luồng màu RGB: 640x480 @ 30 FPS (BGR8) -> /camera/color/image_raw & /camera/image_raw
           + Luồng chiều sâu Depth Map: 640x480 @ 30 FPS (16UC1, đo bằng IR Stereo chính xác từng mm)
             -> /camera/depth/image_raw & /camera/depth/image_rect_raw
           + Căn chỉnh điểm ảnh Align Depth to Color (1:1 pixel mapping): mỗi điểm ảnh (x,y)
             trên ảnh RGB tương ứng chính xác khoảng cách mm trên ảnh Depth.
           + Luồng bản đồ nhiệt màu trực quan: /camera/depth/image_colored (JET Colormap cho RViz2 / rqt)
           + CameraInfo quang học chính xác từ cảm biến: /camera/color/camera_info & /camera/depth/camera_info
           + Ưu điểm: Chiều sâu đo bằng phần cứng quang học ASIC D4, không tốn CPU/GPU, độ trễ gần như 0.
      2. Nếu không có D435: tự động dò tìm iPhone (Iriun Webcam, DroidCam) hoặc USB Webcam ngoài.
  - Hỗ trợ chỉ định trực tiếp:
      --ros-args -p video_device:=auto
      --ros-args -p video_device:=d435
      --ros-args -p video_device:=/dev/video2
      --ros-args -p video_device:=http://172.20.10.1:4747/video
"""

import os
import glob
import time
import threading

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo


class CameraPublisher(Node):
    """Publish ảnh màu và Depth Map từ Intel RealSense D435 hoặc iPhone / USB Webcam."""

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('video_device', 'auto')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('color_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('depth_frame_id', 'camera_depth_optical_frame')
        self.declare_parameter('enable_depth', True)
        self.declare_parameter('align_depth', True)
        self.declare_parameter('publish_camera_info', True)
        self.declare_parameter('publish_depth_colored', True)
        self.declare_parameter('depth_colormap_max_dist', 5.0)  # mét

        self.video_device          = str(self.get_parameter('video_device').value).strip()
        self.width                 = int(self.get_parameter('width').value)
        self.height                = int(self.get_parameter('height').value)
        self.fps                   = float(self.get_parameter('fps').value)
        self.frame_id              = str(self.get_parameter('camera_frame_id').value)
        self.color_frame_id        = str(self.get_parameter('color_frame_id').value) or self.frame_id
        self.depth_frame_id        = str(self.get_parameter('depth_frame_id').value) or self.frame_id
        self.enable_depth          = bool(self.get_parameter('enable_depth').value)
        self.align_depth           = bool(self.get_parameter('align_depth').value)
        self.publish_camera_info   = bool(self.get_parameter('publish_camera_info').value)
        self.publish_depth_colored = bool(self.get_parameter('publish_depth_colored').value)
        self.depth_colormap_max_m  = float(self.get_parameter('depth_colormap_max_dist').value)

        # ── QoS Profile ───────────────────────────────────────────────
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publishers: Color RGB ─────────────────────────────────────
        self.image_pub         = self.create_publisher(Image, '/camera/color/image_raw', cam_qos)
        self.image_alt_pub     = self.create_publisher(Image, '/camera/image_raw', cam_qos)
        self.color_info_pub    = self.create_publisher(CameraInfo, '/camera/color/camera_info', cam_qos)

        # ── Publishers: Depth Map ─────────────────────────────────────
        self.depth_pub         = self.create_publisher(Image, '/camera/depth/image_raw', cam_qos)
        self.depth_rect_pub    = self.create_publisher(Image, '/camera/depth/image_rect_raw', cam_qos)
        self.depth_aligned_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', cam_qos)
        self.depth_info_pub    = self.create_publisher(CameraInfo, '/camera/depth/camera_info', cam_qos)
        self.depth_colored_pub = self.create_publisher(Image, '/camera/depth/image_colored', cam_qos)

        # ── Internal State ────────────────────────────────────────────
        self.is_running         = True
        self._mode              = None    # 'realsense' hoặc 'opencv'
        self._pipeline          = None    # Cho RealSense
        self._align             = None    # Cho RealSense align
        self._spatial_filter    = None    # Lọc làm mịn không gian bảo toàn cạnh
        self._temporal_filter   = None    # Lọc ổn định độ sâu theo thời gian
        self._hole_fill_filter  = None    # Lấp điểm mù quang học (0mm)
        self._depth_scale       = 0.001   # Mét / đơn vị độ sâu (mặc định 1mm)
        self._color_info_msg    = None
        self._depth_info_msg    = None
        self._cap               = None    # Cho OpenCV
        self._frame_count       = 0
        self._last_log          = time.time()
        self._source_name       = "Chưa kết nối"
        self._last_center_depth = 0.0

        self.get_logger().info(
            f"🚀 [CAMERA] Khởi động camera_publisher | Mode config: '{self.video_device}' | "
            f"Kích thước đích: {self.width}x{self.height} @ {int(self.fps)} FPS | "
            f"Depth Map: {'BẬT (IR Stereo 16UC1)' if self.enable_depth else 'TẮT'}"
        )

        # ── Start capture thread ───────────────────────────────────────
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ── RealSense helpers ──────────────────────────────────────────────
    def _open_realsense(self) -> bool:
        """Mở RealSense D435 song song Color stream + Hardware Depth Map stream."""
        if rs is None:
            return False
        try:
            ctx  = rs.context()
            devs = ctx.query_devices()
            if len(devs) == 0:
                return False

            dev      = devs[0]
            name     = dev.get_info(rs.camera_info.name)
            serial   = dev.get_info(rs.camera_info.serial_number)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)

            pipeline = rs.pipeline()
            config   = rs.config()

            # 1. Kích hoạt luồng Màu RGB
            config.enable_stream(
                rs.stream.color,
                self.width, self.height,
                rs.format.bgr8,
                int(self.fps)
            )

            # 2. Kích hoạt luồng Chiều Sâu Depth Map quang học
            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self.width, self.height,
                    rs.format.z16,
                    int(self.fps)
                )

            profile = pipeline.start(config)

            # Lấy thông số quang học thực tế của cảm biến (Intrinsics)
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            c_intr = color_stream.get_intrinsics()

            self._color_info_msg = self._build_camerainfo_msg(
                c_intr, self.width, self.height, self.color_frame_id
            )

            if self.enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                # Đặt Visual Preset = High Density (4) hoặc Medium Density (5) để đo biển báo phẳng ngoài trời không bị thủng lỗ
                try:
                    if depth_sensor.supports(rs.option.visual_preset):
                        depth_sensor.set_option(rs.option.visual_preset, 4.0)  # 4 = High Density
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 1.0)  # Bật IR Laser Projector
                except Exception:
                    pass

                self._depth_scale = float(depth_sensor.get_depth_scale())

                # Khởi tạo bộ lọc hậu xử lý chiều sâu phần cứng của Intel RealSense
                try:
                    self._spatial_filter = rs.spatial_filter()
                    self._spatial_filter.set_option(rs.option.filter_magnitude, 2.0)
                    self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
                    self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20.0)
                    self._temporal_filter = rs.temporal_filter()
                    self._temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
                    self._temporal_filter.set_option(rs.option.filter_smooth_delta, 20.0)
                    self._hole_fill_filter = rs.hole_filling_filter(1)  # Lấp điểm mù từ vùng lân cận
                except Exception:
                    self._spatial_filter = None
                    self._temporal_filter = None
                    self._hole_fill_filter = None

                if self.align_depth:
                    self._align = rs.align(rs.stream.color)
                    # Khi align to color, Depth Map có cùng frame & intrinsics với Color
                    self._depth_info_msg = self._build_camerainfo_msg(
                        c_intr, self.width, self.height, self.depth_frame_id
                    )
                else:
                    self._align = None
                    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                    d_intr = depth_stream.get_intrinsics()
                    self._depth_info_msg = self._build_camerainfo_msg(
                        d_intr, self.width, self.height, self.depth_frame_id
                    )

            # Test đọc 1 frame kiểm tra
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            if not frames.get_color_frame():
                pipeline.stop()
                return False

            self._pipeline = pipeline
            self._mode = 'realsense'
            self._source_name = f"D435 ({name} S/N: {serial} | USB {usb_type})"
            self.get_logger().info(
                f"📷 [REAL_SENSE READY] {self._source_name}\n"
                f"   • Luồng 1 (RGB):   {self.width}x{self.height} @ {int(self.fps)} FPS (BGR8)\n"
                f"   • Luồng 2 (Depth): {self.width}x{self.height} @ {int(self.fps)} FPS (16UC1 IR Stereo - mm)\n"
                f"   • Align to Color:  {'BẬT (khớp 1:1 từng pixel)' if self.align_depth else 'TẮT'}\n"
                f"   • Depth Scale:     {self._depth_scale:.6f} m/unit"
            )
            return True

        except Exception as e:
            if self._pipeline:
                try: self._pipeline.stop()
                except Exception: pass
                self._pipeline = None
            return False

    def _close_realsense(self):
        if self._pipeline:
            try: self._pipeline.stop()
            except Exception: pass
            self._pipeline = None

    def _build_camerainfo_msg(self, intr, width, height, frame_id) -> CameraInfo:
        """Tạo message CameraInfo chuẩn ROS 2 từ RealSense intrinsics."""
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = 'plumb_bob'
        msg.d = [float(x) for x in intr.coeffs] if hasattr(intr, 'coeffs') else [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = float(intr.fx)
        fy = float(intr.fy)
        cx = float(intr.ppx)
        cy = float(intr.ppy)
        msg.k = [fx,  0.0, cx,
                 0.0, fy,  cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx,  0.0, cx,  0.0,
                 0.0, fy,  cy,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def _build_approx_camerainfo(self, width, height, frame_id) -> CameraInfo:
        """Tạo CameraInfo ước lượng cho webcam thường / iPhone."""
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = float(width) * 0.8
        fy = float(width) * 0.8
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        msg.k = [fx,  0.0, cx,
                 0.0, fy,  cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx,  0.0, cx,  0.0,
                 0.0, fy,  cy,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    # ── OpenCV / V4L2 / iPhone helpers ─────────────────────────────────
    def _find_best_v4l2_device(self):
        """Tự động tìm device iPhone (Iriun, Droidcam) hoặc webcam rời."""
        devices = []
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            name_file = os.path.join(path, "name")
            name = ""
            if os.path.exists(name_file):
                try:
                    with open(name_file, 'r') as f:
                        name = f.read().strip()
                except Exception:
                    pass
            dev_node = "/dev/" + os.path.basename(path)
            devices.append((dev_node, name))

        if not devices:
            return None, "Không tìm thấy thiết bị video"

        # 1. Ưu tiên Iriun, Droidcam, iPhone
        for dev_node, name in devices:
            n_lower = name.lower()
            if any(k in n_lower for k in ['iriun', 'droidcam', 'iphone', 'phone', 'apple']):
                return dev_node, f"iPhone ({name} tại {dev_node})"

        # 2. Ưu tiên thiết bị không phải webcam tích hợp laptop (thường là HD User Facing)
        for dev_node, name in devices:
            n_lower = name.lower()
            if not any(k in n_lower for k in ['user facing', 'integrated', 'built-in']):
                return dev_node, f"Webcam ({name} tại {dev_node})"

        # 3. Fallback lấy device đầu tiên
        return devices[0][0], f"Camera ({devices[0][1]} tại {devices[0][0]})"

    def _open_opencv(self, target_dev) -> bool:
        """Mở camera qua OpenCV (hỗ trợ V4L2 device index, path, hoặc URL)."""
        if cv2 is None:
            self.get_logger().error("❌ Chưa cài đặt thư viện OpenCV (cv2)!")
            return False

        try:
            target_str = str(target_dev).strip()
            cap_target = target_str

            # Tự động nhận diện nếu người dùng truyền IP DroidCam (vd: 192.168.1.5 hoặc 10.10.160.14:4747)
            ip_candidate = target_str.split(':')[0]
            if ip_candidate.count('.') == 3 and all(p.isdigit() for p in ip_candidate.split('.')):
                if not target_str.startswith(('http://', 'https://', 'rtsp://')):
                    if ':' in target_str:
                        cap_target = f"http://{target_str}/video"
                    else:
                        cap_target = f"http://{target_str}:4747/video"
            elif target_str.startswith('/dev/video'):
                try:
                    cap_target = int(target_str.replace('/dev/video', ''))
                except ValueError:
                    cap_target = target_str
            elif target_str.isdigit():
                cap_target = int(target_str)

            # Dùng V4L2 backend cho Linux device
            if isinstance(cap_target, int):
                cap = cv2.VideoCapture(cap_target, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(cap_target)

            if not cap.isOpened():
                cap.release()
                return False

            # Cấu hình độ phân giải và FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm đệm để không bị trễ khung hình

            # Test đọc 1 frame
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                cap.release()
                return False

            self._cap = cap
            self._mode = 'opencv'
            self._color_info_msg = self._build_approx_camerainfo(
                self.width, self.height, self.color_frame_id
            )
            self._source_name = f"{cap_target} ({frame.shape[1]}x{frame.shape[0]})"
            self.get_logger().info(
                f"📷 [IPHONE / WEBCAM READY] Nguồn: {self._source_name} | "
                f"Đích: {self.width}x{self.height} @ {int(self.fps)} FPS | BGR8"
            )
            return True

        except Exception:
            if self._cap:
                try: self._cap.release()
                except Exception: pass
                self._cap = None
            return False

    def _close_opencv(self):
        if self._cap:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    def _connect_camera(self) -> bool:
        """Thử kết nối camera theo thứ tự ưu tiên tối ưu."""
        dev_lower = self.video_device.lower()

        # 0. Chế độ chọn đích danh DroidCam / iPhone
        if dev_lower in ['droidcam', 'iphone', 'phone']:
            test_candidates = [
                'http://172.20.10.1:4747/video',    # iPhone USB Personal Hotspot
                'http://127.0.0.1:4747/video',     # Local droidcam-cli loopback
            ]
            for u in test_candidates:
                if self._open_opencv(u):
                    return True
            best_dev, _ = self._find_best_v4l2_device()
            if best_dev and self._open_opencv(best_dev):
                return True
            return False

        # 1. ƯU TIÊN 1 HÀNG ĐẦU: RealSense D435 cắm qua USB (khi chọn auto, d435, hoặc mặc định /dev/video0)
        # Tránh lỗi launch file truyền /dev/video0 làm mở nhầm webcam laptop và mất kênh Depth D435!
        if self.video_device in ['auto', 'd435', 'realsense', '/dev/video0']:
            if self._open_realsense():
                return True
            if self.video_device in ['d435', 'realsense']:
                return False

        # 2. Chế độ chỉ định URL, IP hoặc device cụ thể khác
        if self.video_device not in ['auto', 'd435', 'realsense']:
            return self._open_opencv(self.video_device)

        # 3. Ưu tiên 2: Kết nối iPhone DroidCam IP tĩnh 172.20.10.1 (Cáp USB / Hotspot)
        if self._open_opencv('http://172.20.10.1:4747/video'):
            return True

        # 4. Tự động dò iPhone / V4L2 device ngoài
        best_dev, dev_desc = self._find_best_v4l2_device()
        if best_dev and self._open_opencv(best_dev):
            return True

        # 5. Thử các video device khác trên máy
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            dev_node = "/dev/" + os.path.basename(path)
            if self._open_opencv(dev_node):
                return True

        return False

    # ── Main capture loop ──────────────────────────────────────────────
    def _capture_loop(self):
        consecutive_fail = 0

        while self.is_running and rclpy.ok():
            # Nếu chưa kết nối camera
            if self._mode is None:
                if not self._connect_camera():
                    self.get_logger().warn(
                        "⏳ Đang tìm camera (Ưu tiên Intel RealSense D435 hoặc iPhone qua DroidCam/USB)...",
                        throttle_duration_sec=3.0
                    )
                    time.sleep(1.5)
                    continue
                consecutive_fail = 0

            color_frame_data = None
            depth_frame_data = None

            # 1. Đọc từ RealSense D435 (Dual stream: RGB + Depth)
            if self._mode == 'realsense':
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                    if self._align:
                        frames = self._align.process(frames)

                    c_frame = frames.get_color_frame()
                    d_frame = frames.get_depth_frame() if self.enable_depth else None

                    if not c_frame:
                        raise RuntimeError("No color frame from RealSense")

                    color_frame_data = np.asanyarray(c_frame.get_data())
                    if d_frame:
                        # Áp dụng bộ lọc khử nhiễu không gian, thời gian & lấp điểm mù trên mặt phẳng biển mũi tên
                        if self._spatial_filter is not None:
                            d_frame = self._spatial_filter.process(d_frame)
                        if self._temporal_filter is not None:
                            d_frame = self._temporal_filter.process(d_frame)
                        if self._hole_fill_filter is not None:
                            d_frame = self._hole_fill_filter.process(d_frame)

                        raw_d = np.asanyarray(d_frame.get_data())
                        # Chuẩn hóa tỷ lệ tuyệt đối về đơn vị milimét (16UC1 = 1mm/unit) theo depth_scale phần cứng
                        scale_mm = self._depth_scale * 1000.0
                        if abs(scale_mm - 1.0) > 1e-3:
                            depth_frame_data = np.clip(np.round(raw_d.astype(np.float32) * scale_mm), 0, 65535).astype(np.uint16)
                        else:
                            depth_frame_data = raw_d

                    # Giải phóng tham chiếu frame ngay để librealsense tái sử dụng buffer
                    del c_frame, d_frame, frames
                    consecutive_fail = 0

                except Exception:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        self.get_logger().warn(
                            "⚠️ Mất tín hiệu RealSense D435, đang kết nối lại...",
                            throttle_duration_sec=5.0
                        )
                        self._close_realsense()
                        self._mode = None
                    time.sleep(0.05)
                    continue

            # 2. Đọc từ OpenCV (iPhone / USB Webcam ngoài)
            elif self._mode == 'opencv':
                try:
                    ret, raw_frame = self._cap.read()
                    if not ret or raw_frame is None or raw_frame.size == 0:
                        raise RuntimeError("Failed to grab frame from OpenCV")

                    fh, fw = raw_frame.shape[:2]
                    if fw != self.width or fh != self.height:
                        color_frame_data = cv2.resize(raw_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    else:
                        color_frame_data = raw_frame

                    consecutive_fail = 0
                except Exception:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        self.get_logger().warn(
                            "⚠️ Mất tín hiệu iPhone / Webcam, đang kết nối lại...",
                            throttle_duration_sec=5.0
                        )
                        self._close_opencv()
                        self._mode = None
                    time.sleep(0.05)
                    continue

            if color_frame_data is None:
                continue

            now_stamp = self.get_clock().now().to_msg()

            # ── Publish RGB Image ──────────────────────────────────────
            ch, cw = color_frame_data.shape[:2]
            color_msg = Image()
            color_msg.header.stamp    = now_stamp
            color_msg.header.frame_id = self.color_frame_id
            color_msg.height          = ch
            color_msg.width           = cw
            color_msg.encoding        = 'bgr8'
            color_msg.is_bigendian    = False
            color_msg.step            = cw * 3
            color_msg.data            = color_frame_data.tobytes()

            if not self.is_running or not rclpy.ok():
                break

            try:
                self.image_pub.publish(color_msg)
                self.image_alt_pub.publish(color_msg)
                if self.publish_camera_info and self._color_info_msg is not None:
                    self._color_info_msg.header.stamp = now_stamp
                    self.color_info_pub.publish(self._color_info_msg)
            except Exception:
                break

            # ── Publish Depth Map (16UC1 - mm) ─────────────────────────
            if depth_frame_data is not None:
                dh, dw = depth_frame_data.shape[:2]
                depth_msg = Image()
                depth_msg.header.stamp    = now_stamp
                depth_msg.header.frame_id = self.depth_frame_id
                depth_msg.height          = dh
                depth_msg.width           = dw
                depth_msg.encoding        = '16UC1'
                depth_msg.is_bigendian    = False
                depth_msg.step            = dw * 2
                depth_msg.data            = depth_frame_data.tobytes()

                try:
                    self.depth_pub.publish(depth_msg)
                    self.depth_rect_pub.publish(depth_msg)
                    self.depth_aligned_pub.publish(depth_msg)
                    if self.publish_camera_info and self._depth_info_msg is not None:
                        self._depth_info_msg.header.stamp = now_stamp
                        self.depth_info_pub.publish(self._depth_info_msg)
                except Exception:
                    break

                # ── Publish Colorized Depth Map (JET Colormap trực quan) ─
                if self.publish_depth_colored and cv2 is not None:
                    try:
                        max_mm = max(1000.0, self.depth_colormap_max_m * 1000.0)
                        depth_scaled = cv2.convertScaleAbs(depth_frame_data, alpha=255.0 / max_mm)
                        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
                        depth_colored[depth_frame_data == 0] = [0, 0, 0]  # Điểm mù hiển thị đen

                        col_msg = Image()
                        col_msg.header.stamp    = now_stamp
                        col_msg.header.frame_id = self.depth_frame_id
                        col_msg.height          = dh
                        col_msg.width           = dw
                        col_msg.encoding        = 'bgr8'
                        col_msg.is_bigendian    = False
                        col_msg.step            = dw * 3
                        col_msg.data            = depth_colored.tobytes()
                        self.depth_colored_pub.publish(col_msg)
                    except Exception:
                        pass

                # Lưu khoảng cách tâm ảnh để log
                self._last_center_depth = float(depth_frame_data[dh // 2, dw // 2])

            # ── Thống kê định kỳ mỗi 15 giây ──────────────────────────
            self._frame_count += 1
            now = time.time()
            if now - self._last_log >= 15.0:
                fps_actual = self._frame_count / (now - self._last_log)
                depth_str = f" | Tâm Depth: {self._last_center_depth:.0f} mm" if depth_frame_data is not None else ""
                self.get_logger().info(
                    f"📷 [{self._source_name}] {cw}x{ch} | {fps_actual:.1f} FPS"
                    f"{depth_str} | {self._frame_count} frames ➔ AI CNN / Depth"
                )
                self._frame_count = 0
                self._last_log = now

    def destroy_node(self):
        self.is_running = False
        self._close_realsense()
        self._close_opencv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()
