#!/usr/bin/env python3
"""
Camera Publisher Node — Hỗ trợ Intel RealSense D435 (ưu tiên) + USB Webcam + iPhone DroidCam.

Ưu tiên kết nối:
  1. Intel RealSense D435 qua pyrealsense2 (Color stream RGB, KHÔNG phải IR/Depth)
  2. USB Webcam qua OpenCV V4L2 (MJPG/YUYV)
  3. iPhone DroidCam / IP Camera qua HTTP stream

Chức năng:
  - Publish ảnh màu 640×480 lên /camera/color/image_raw (encoding bgr8)
  - Background thread capture riêng để không block ROS 2 callbacks
  - Tự động reconnect khi camera bị ngắt

Sử dụng:
  ros2 run my_robot_bringup camera_publisher
  ros2 run my_robot_bringup camera_publisher --ros-args -p width:=640 -p height:=480 -p fps:=30.0
"""

import os
import time
import threading
import socket
from urllib.parse import urlparse

# Set OpenCV FFMPEG network timeout (1.5s) to avoid hanging on dead/unreachable network streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;1500000|stimeout;1500000"

import numpy as np

try:
    import cv2
except ImportError:
    print("❌ Thiếu thư viện opencv-python! Chạy: sudo apt install python3-opencv")
    cv2 = None

# ── Thử import pyrealsense2 ──────────────────────────────────────────
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    rs = None
    _HAS_REALSENSE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


def _is_url_accessible(url: str, timeout: float = 0.8) -> bool:
    """Kiểm tra nhanh xem IP/port của stream có phản hồi không (tránh cv2.VideoCapture bị block 30s)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (80 if parsed.scheme == 'http' else 443)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class CameraPublisher(Node):
    """Publish ảnh từ RealSense D435 / USB Webcam / iPhone DroidCam."""

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('camera_frame_id', 'camera_link')

        self.device = self.get_parameter('video_device').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.frame_id = self.get_parameter('camera_frame_id').value

        # ── Publisher (RELIABLE để tương thích CNN driver + wifi_cam_bridge) ─
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_pub = self.create_publisher(
            Image, '/camera/color/image_raw', cam_qos)
        # Fallback topic cho compatibility
        self.image_alt_pub = self.create_publisher(
            Image, '/camera/image_raw', cam_qos)

        # ── State ─────────────────────────────────────────────────────
        self.is_running = True
        self.cap = None                # OpenCV VideoCapture (USB/iPhone)
        self._rs_pipeline = None       # RealSense pipeline
        self._rs_align = None          # RealSense alignment
        self._camera_mode = None       # 'realsense' | 'opencv'
        self._frame_count = 0
        self._last_log_time = time.time()

        # ── Start capture & auto-reconnect thread ─────────────────────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    # ══════════════════════════════════════════════════════════════════
    # RealSense D435 — pyrealsense2
    # ══════════════════════════════════════════════════════════════════
    def _try_open_realsense(self):
        """Thử mở Intel RealSense D435 qua pyrealsense2. Trả về True nếu thành công."""
        if not _HAS_REALSENSE:
            return False
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return False

            dev_name = devices[0].get_info(rs.camera_info.name)
            serial = devices[0].get_info(rs.camera_info.serial_number)

            pipeline = rs.pipeline()
            config = rs.config()

            # Chỉ bật Color stream (RGB) — KHÔNG bật Depth/IR
            config.enable_stream(
                rs.stream.color,
                self.width, self.height,
                rs.format.bgr8,
                int(self.fps)
            )

            profile = pipeline.start(config)

            # Đọc thử 1 frame để xác nhận
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                pipeline.stop()
                return False

            actual_w = color_frame.get_width()
            actual_h = color_frame.get_height()

            self._rs_pipeline = pipeline
            self._camera_mode = 'realsense'

            self.get_logger().info(
                f"📷 [REALSENSE D435 ĐÃ KẾT NỐI] {dev_name} (S/N: {serial}) | "
                f"Color Stream: {actual_w}x{actual_h} @ {int(self.fps)} FPS | "
                f"🟢 Ảnh MÀU (RGB) — Không phải IR")
            return True

        except Exception as e:
            self.get_logger().warn(
                f"⚠️ RealSense D435 không thể mở: {e}", throttle_duration_sec=5.0)
            if self._rs_pipeline is not None:
                try:
                    self._rs_pipeline.stop()
                except Exception:
                    pass
                self._rs_pipeline = None
            return False

    def _read_realsense_frame(self):
        """Đọc 1 frame màu từ RealSense pipeline. Trả về numpy BGR array hoặc None."""
        try:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            # Chuyển sang numpy array (đã là BGR8)
            frame = np.asanyarray(color_frame.get_data())
            return frame
        except Exception:
            return None

    def _stop_realsense(self):
        """Dừng RealSense pipeline an toàn."""
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None
        self._camera_mode = None

    # ══════════════════════════════════════════════════════════════════
    # OpenCV — USB Webcam / iPhone DroidCam
    # ══════════════════════════════════════════════════════════════════
    def _try_open_urls(self, urls):
        """Thử mở luồng mạng (iPhone/IP camera). Trả về VideoCapture nếu thành công."""
        for url in urls:
            if not _is_url_accessible(url, timeout=0.8):
                continue
            for api in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
                cap = None
                try:
                    cap = cv2.VideoCapture(url, api)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            self.get_logger().info(
                                f"📱 Camera iPhone / IP Stream đã kết nối thành công: {url} "
                                f"({actual_w}x{actual_h})")
                            return cap
                        cap.release()
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
        return None

    def _get_device_name(self, dev):
        """Lấy tên phần cứng của thiết bị video từ sysfs."""
        try:
            v_id = str(dev).replace('/dev/', '')
            name_file = f"/sys/class/video4linux/{v_id}/name"
            if os.path.exists(name_file):
                with open(name_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return ""

    def _try_open_usb_devices(self):
        """
        Thử mở camera USB (/dev/video*) — BỎ QUA các cổng RealSense.
        RealSense được xử lý riêng qua pyrealsense2.
        """
        raw_candidates = []
        if isinstance(self.device, str) and self.device.startswith('/dev/video') and os.path.exists(self.device):
            raw_candidates.append(self.device)
        for i in range(12):
            dev = f'/dev/video{i}'
            if dev not in raw_candidates and os.path.exists(dev):
                raw_candidates.append(dev)

        # Lọc bỏ RealSense (đã xử lý riêng qua pyrealsense2)
        standard_cams = []
        for dev in raw_candidates:
            dev_name = self._get_device_name(dev).lower()
            if 'realsense' in dev_name:
                continue  # Bỏ qua RealSense — dùng pyrealsense2
            standard_cams.append((dev, self._get_device_name(dev)))

        # Thử các định dạng: MJPG, YUYV và Default
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'MJPG'),
            cv2.VideoWriter_fourcc(*'YUYV'),
            0  # Default format
        ]

        for dev, name in standard_cams:
            if dev.startswith('/dev/video') and dev.replace('/dev/video', '').isdigit():
                dev_id = int(dev.replace('/dev/video', ''))
            else:
                dev_id = dev

            # Thử mở trực tiếp qua V4L2
            cap = None
            try:
                cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
                if not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue

                # Cấu hình định dạng và độ phân giải
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for fourcc in fourcc_options:
                    if fourcc != 0:
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)

                    ret, test_frame = cap.read()
                    # Kiểm tra frame hợp lệ và là ảnh màu 3 kênh (BGR)
                    if ret and test_frame is not None and len(test_frame.shape) == 3 and test_frame.shape[2] == 3:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        self.get_logger().info(
                            f"📷 [USB WEBCAM ĐÃ KẾT NỐI] {name} tại {dev}: "
                            f"{actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
                        return cap

                cap.release()
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return None

    def _open_camera(self):
        """
        Mở camera thông minh theo thứ tự ưu tiên:
        1. Intel RealSense D435 qua pyrealsense2 (Color RGB, ảnh MÀU)
        2. USB Webcam qua OpenCV V4L2
        3. iPhone DroidCam / IP Camera qua HTTP stream
        """
        urls_to_try = []
        is_http_target = isinstance(self.device, str) and self.device.startswith(('http://', 'https://', 'rtsp://'))

        if is_http_target:
            urls_to_try.append(self.device)
            if self.device.endswith('/video'):
                urls_to_try.append(self.device.replace('/video', '/mjpegfeed'))
            elif self.device.endswith('/mjpegfeed'):
                urls_to_try.append(self.device.replace('/mjpegfeed', '/video'))

        # Danh sách IP phổ biến của iPhone DroidCam
        default_iphones = [
            'http://172.20.10.1:4747/video',      # iPhone Hotspot mặc định
            'http://127.0.0.1:4747/video',       # DroidCam qua cáp USB tethering
            'http://192.168.1.100:4747/video',   # WiFi gia đình / phòng thí nghiệm
            'http://192.168.43.1:4747/video',    # Hotspot phụ
        ]
        for u in default_iphones:
            if u not in urls_to_try:
                urls_to_try.append(u)

        # 1. ƯU TIÊN SỐ 1: Intel RealSense D435 qua pyrealsense2
        if self._try_open_realsense():
            return None  # RealSense dùng pipeline riêng, không cần VideoCapture

        # 2. USB Webcam (bỏ qua cổng RealSense V4L2)
        if not is_http_target:
            cap = self._try_open_usb_devices()
            if cap is not None:
                self._camera_mode = 'opencv'
                return cap

        # 3. iPhone DroidCam / IP Camera
        cap = self._try_open_urls(urls_to_try)
        if cap is not None:
            self._camera_mode = 'opencv'
            return cap

        # 4. Fallback: USB Webcam (nếu target là HTTP nhưng không kết nối được)
        if is_http_target:
            cap = self._try_open_usb_devices()
            if cap is not None:
                self._camera_mode = 'opencv'
                return cap

        return None

    # ══════════════════════════════════════════════════════════════════
    # Main capture loop
    # ══════════════════════════════════════════════════════════════════
    def _capture_loop(self):
        """Background thread: đọc frame liên tục và tự động kết nối lại nếu camera bị ngắt."""
        null_fd = None
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
        except Exception:
            pass

        consecutive_failures = 0
        try:
            while self.is_running:
                # ── Kiểm tra kết nối ──────────────────────────────────
                if self._camera_mode is None:
                    self.cap = self._open_camera()
                    if self._camera_mode is None:
                        self.get_logger().warn(
                            f"⏳ Đang dò tìm & kết nối Camera ({self.device})... "
                            f"(RealSense D435 / USB Webcam / DroidCam)",
                            throttle_duration_sec=4.0)
                        time.sleep(1.5)
                        continue
                    consecutive_failures = 0

                # ── Đọc frame ─────────────────────────────────────────
                frame = None

                if self._camera_mode == 'realsense':
                    frame = self._read_realsense_frame()
                elif self._camera_mode == 'opencv' and self.cap is not None:
                    # Tạm redirect stderr để suppress libjpeg warnings
                    old_err = None
                    if null_fd is not None:
                        try:
                            old_err = os.dup(2)
                            os.dup2(null_fd, 2)
                        except Exception:
                            old_err = None

                    ret, frame = self.cap.read()
                    if not ret:
                        frame = None

                    # Khôi phục stderr
                    if old_err is not None:
                        try:
                            os.dup2(old_err, 2)
                            os.close(old_err)
                        except Exception:
                            pass

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 150:
                        self.get_logger().warn(
                            "Mất tín hiệu camera quá 3 giây, đang tự động kết nối lại...",
                            throttle_duration_sec=5.0)
                        self._cleanup_camera()
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0

                # ── Resize & publish ──────────────────────────────────
                if frame.shape[1] != 640 or frame.shape[0] != 480:
                    pub_frame = cv2.resize(frame, (640, 480),
                                           interpolation=cv2.INTER_LINEAR)
                else:
                    pub_frame = frame

                ph, pw = pub_frame.shape[:2]
                pc = pub_frame.shape[2] if len(pub_frame.shape) == 3 else 1

                # Tạo ROS 2 Image message
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.frame_id
                msg.height = ph
                msg.width = pw
                msg.encoding = 'bgr8'
                msg.is_bigendian = False
                msg.step = pw * pc
                msg.data = pub_frame.tobytes()

                if not self.is_running or not rclpy.ok():
                    break
                try:
                    self.image_pub.publish(msg)
                    self.image_alt_pub.publish(msg)
                except Exception:
                    break

                # Log thống kê mỗi 30 giây
                self._frame_count += 1
                now = time.time()
                if now - self._last_log_time >= 30.0:
                    elapsed = now - self._last_log_time if self._last_log_time > 0 else 30.0
                    fps_actual = self._frame_count / elapsed if elapsed > 0 else 0
                    src_tag = "RealSense D435" if self._camera_mode == 'realsense' else "OpenCV"
                    self.get_logger().info(
                        f"📷 [{src_tag}] {pw}x{ph} | {fps_actual:.1f} FPS | "
                        f"Đang cấp dữ liệu trực tiếp cho AI CNN ({self._frame_count} frames)")
                    self._frame_count = 0
                    self._last_log_time = now

        finally:
            if null_fd is not None:
                try:
                    os.close(null_fd)
                except Exception:
                    pass

    def _cleanup_camera(self):
        """Dọn dẹp tất cả kết nối camera."""
        if self._camera_mode == 'realsense':
            self._stop_realsense()
        elif self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._camera_mode = None

    def destroy_node(self):
        self.is_running = False
        self._cleanup_camera()
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
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
