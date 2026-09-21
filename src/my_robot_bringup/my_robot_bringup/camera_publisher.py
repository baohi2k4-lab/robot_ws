#!/usr/bin/env python3
"""
Camera Publisher Node — Thu hình 1080p@60fps từ USB Webcam bằng OpenCV MJPG.

Thay thế hoàn toàn v4l2_camera_node (không hỗ trợ giải mã MJPG trên Jazzy).
OpenCV giải mã MJPG chuẩn xác trên mọi nền tảng (Pi 4/5, x86 PC).

Chức năng:
  - Mở camera USB (DV20) bằng cv2.VideoCapture + CAP_V4L2 + MJPG fourcc
  - Publish ảnh gốc 1920×1080 lên /camera/color/image_raw (encoding rgb8)
  - Background thread capture riêng để không block ROS 2 callbacks

Sử dụng:
  ros2 run my_robot_bringup camera_publisher
  ros2 run my_robot_bringup camera_publisher --ros-args -p width:=1920 -p height:=1080 -p fps:=60.0
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
    """Publish ảnh USB Webcam hoặc iPhone DroidCam qua OpenCV MJPG backend."""

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 60.0)
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
        self.cap = None
        self._frame_count = 0
        self._last_log_time = time.time()

        # ── Start capture & auto-reconnect thread ─────────────────────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()

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
        Thử mở camera USB (/dev/video*).
        Đặc biệt tối ưu cho Intel RealSense D435:
          - Tự động nhận diện và ưu tiên cổng RGB Color Camera.
          - Bỏ qua các cổng Depth/IR/Metadata để tránh nhận nhầm ảnh hồng ngoại.
          - Thử cả định dạng MJPG và YUYV/Default để tương thích 100% với D435.
        """
        raw_candidates = []
        if isinstance(self.device, str) and self.device.startswith('/dev/video') and os.path.exists(self.device):
            raw_candidates.append(self.device)
        for i in range(12):
            dev = f'/dev/video{i}'
            if dev not in raw_candidates and os.path.exists(dev):
                raw_candidates.append(dev)

        # Phân loại và xếp thứ tự ưu tiên thiết bị
        realsense_cams = []
        standard_cams = []

        for dev in raw_candidates:
            dev_name = self._get_device_name(dev).lower()
            if 'realsense' in dev_name:
                realsense_cams.append((dev, self._get_device_name(dev)))
            else:
                standard_cams.append((dev, self._get_device_name(dev)))

        # Danh sách ưu tiên: RealSense D435 -> Các USB Webcam khác
        sorted_candidates = realsense_cams + standard_cams

        # Thử các định dạng: MJPG, UYVY (gốc D435 RGB), YUYV và Default
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'MJPG'),
            cv2.VideoWriter_fourcc(*'UYVY'),
            cv2.VideoWriter_fourcc(*'YUYV'),
            0  # Default format
        ]

        for dev, name in sorted_candidates:
            if dev.startswith('/dev/video') and dev.replace('/dev/video', '').isdigit():
                dev_id = int(dev.replace('/dev/video', ''))
            else:
                dev_id = dev

            is_d435 = 'realsense' in name.lower()

            # Thử mở trực tiếp qua V4L2
            cap = None
            try:
                cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
                if not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue  # Bỏ qua ngay lập tức cổng không mở được (ví dụ Depth/IR)

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
                        dev_tag = "Intel RealSense D435 (RGB)" if is_d435 else f"USB Camera ({name})"
                        self.get_logger().info(
                            f"📷 [CAMERA ĐÃ KẾT NỐI] {dev_tag} tại {dev}: "
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
        Mở camera thông minh: Tự động nhận diện 'Ghim cái nào xài cái đó'.
        Hỗ trợ cả Intel RealSense D435 và iPhone (DroidCam / IP Stream).
        Tự động chuyển đổi mượt mà khi cắm / rút thiết bị.
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

        # 1. Nếu người dùng chỉ định rõ là URL HTTP thì ưu tiên thử iPhone trước
        if is_http_target:
            cap = self._try_open_urls(urls_to_try)
            if cap is not None:
                return cap
            # Nếu iPhone không mở được, tự động tìm RealSense / USB Webcam
            cap = self._try_open_usb_devices()
            if cap is not None:
                return cap

        # 2. Mặc định: Ưu tiên tìm Intel RealSense D435 / USB Webcam cắm trực tiếp
        cap = self._try_open_usb_devices()
        if cap is not None:
            return cap

        # 3. Nếu không có RealSense / USB cắm máy, tự động dò tìm luồng iPhone DroidCam
        cap = self._try_open_urls(urls_to_try)
        if cap is not None:
            return cap

        return None

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
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self._open_camera()
                    if self.cap is None or not self.cap.isOpened():
                        self.get_logger().warn(
                            f"⏳ Đang dò tìm & kết nối Camera ({self.device})... "
                            f"(Kiểm tra DroidCam trên iPhone hoặc cắm lại cổng USB Webcam)",
                            throttle_duration_sec=4.0)
                        time.sleep(1.5)
                        continue
                    consecutive_failures = 0
                # Tạm redirect stderr để suppress libjpeg warnings
                old_err = None
                if null_fd is not None:
                    try:
                        old_err = os.dup(2)
                        os.dup2(null_fd, 2)
                    except Exception:
                        old_err = None

                ret, frame = self.cap.read()

                # Khôi phục stderr
                if old_err is not None:
                    try:
                        os.dup2(old_err, 2)
                        os.close(old_err)
                    except Exception:
                        pass

                if not ret or frame is None:
                    consecutive_failures += 1
                    # Cho phép chịu lỗi liên tiếp trong ~3 giây (150 lần x 0.02s) trước khi hủy kết nối
                    if consecutive_failures >= 150:
                        self.get_logger().warn("Mất tín hiệu camera quá 3 giây, đang tự động kết nối lại...", throttle_duration_sec=5.0)
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0

                # Resize 1080p → 640x480 trước khi publish
                # (CNN driver resize lại 512x512, wifi_cam_bridge resize 320x240)
                # Tiết kiệm từ 6.2MB → 920KB mỗi frame
                pub_frame = cv2.resize(frame, (640, 480),
                                       interpolation=cv2.INTER_LINEAR)

                ph, pw, pc = pub_frame.shape

                # Tạo ROS 2 Image message (giữ bgr8 — không cần chuyển đổi)
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

                # Log thống kê mỗi 30 giây (giảm tải màn hình terminal)
                self._frame_count += 1
                now = time.time()
                if now - self._last_log_time >= 30.0:
                    elapsed = now - self._last_log_time if self._last_log_time > 0 else 30.0
                    fps_actual = self._frame_count / elapsed if elapsed > 0 else 0
                    self.get_logger().info(
                        f"📷 Camera: {pw}x{ph} | {fps_actual:.1f} FPS | "
                        f"Đang cấp dữ liệu trực tiếp cho AI CNN ({self._frame_count} frames)")
                    self._frame_count = 0
                    self._last_log_time = now

        finally:
            if null_fd is not None:
                try:
                    os.close(null_fd)
                except Exception:
                    pass

    def destroy_node(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
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
