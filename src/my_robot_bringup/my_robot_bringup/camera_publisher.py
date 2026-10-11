#!/usr/bin/env python3
"""
Camera Publisher Node — Hỗ trợ Intel RealSense D435 và iPhone / USB Webcam / RTSP Stream.

Tính năng:
  - Tự động nhận diện (video_device='auto'):
      1. Ưu tiên RealSense D435 nếu đang cắm.
      2. Nếu không có D435: tự động dò tìm iPhone (Iriun Webcam, DroidCam) hoặc USB Webcam ngoài.
  - Hỗ trợ chỉ định trực tiếp:
      --ros-args -p video_device:=/dev/video2
      --ros-args -p video_device:=http://192.168.1.50:8080/video
  - Chuẩn hóa đầu ra BGR8, kích thước chuẩn (mặc định 640x480) lên:
      /camera/color/image_raw
      /camera/image_raw
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
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    """Publish ảnh màu từ Intel RealSense D435 hoặc iPhone / USB Webcam."""

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('video_device', 'auto')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('camera_frame_id', 'camera_link')

        self.video_device = str(self.get_parameter('video_device').value).strip()
        self.width        = int(self.get_parameter('width').value)
        self.height       = int(self.get_parameter('height').value)
        self.fps          = float(self.get_parameter('fps').value)
        self.frame_id     = str(self.get_parameter('camera_frame_id').value)

        # ── Publisher ─────────────────────────────────────────────────
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_pub     = self.create_publisher(Image, '/camera/color/image_raw', cam_qos)
        self.image_alt_pub = self.create_publisher(Image, '/camera/image_raw', cam_qos)

        # ── State ─────────────────────────────────────────────────────
        self.is_running    = True
        self._mode         = None    # 'realsense' hoặc 'opencv'
        self._pipeline     = None    # Cho RealSense
        self._cap          = None    # Cho OpenCV
        self._frame_count  = 0
        self._last_log     = time.time()
        self._source_name  = "Chưa kết nối"

        self.get_logger().info(
            f"🚀 [CAMERA] Khởi động camera_publisher | Mode config: '{self.video_device}' | "
            f"Kích thước đích: {self.width}x{self.height} @ {int(self.fps)} FPS"
        )

        # ── Start capture thread ───────────────────────────────────────
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ── RealSense helpers ──────────────────────────────────────────────
    def _open_realsense(self) -> bool:
        """Mở RealSense D435 Color stream."""
        if rs is None:
            return False
        try:
            ctx  = rs.context()
            devs = ctx.query_devices()
            if len(devs) == 0:
                return False

            name   = devs[0].get_info(rs.camera_info.name)
            serial = devs[0].get_info(rs.camera_info.serial_number)

            pipeline = rs.pipeline()
            config   = rs.config()
            config.enable_stream(
                rs.stream.color,
                self.width, self.height,
                rs.format.bgr8,
                int(self.fps)
            )
            pipeline.start(config)

            # Test đọc 1 frame
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            if not frames.get_color_frame():
                pipeline.stop()
                return False

            self._pipeline = pipeline
            self._mode = 'realsense'
            self._source_name = f"D435 ({name} S/N: {serial})"
            self.get_logger().info(
                f"📷 [REAL_SENSE READY] {self._source_name} | "
                f"{self.width}x{self.height} @ {int(self.fps)} FPS | Color BGR8")
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

        # 2. Ưu tiên thiết bị không phải webcam tích hợp laptop (thường là video0/video1 HD User Facing)
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
            self._source_name = f"{cap_target} ({frame.shape[1]}x{frame.shape[0]})"
            self.get_logger().info(
                f"📷 [IPHONE / WEBCAM READY] Nguồn: {self._source_name} | "
                f"Đích: {self.width}x{self.height} @ {int(self.fps)} FPS | BGR8"
            )
            return True

        except Exception as e:
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
        """Thử kết nối camera theo cấu hình video_device."""
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

        # 1. Chế độ chỉ định URL, IP hoặc device cụ thể
        if self.video_device not in ['auto', 'd435']:
            return self._open_opencv(self.video_device)

        # 2. Ưu tiên 1: Kết nối trực tiếp iPhone DroidCam IP tĩnh 172.20.10.1 (Cáp USB / Hotspot)
        if self.video_device != 'd435':
            if self._open_opencv('http://172.20.10.1:4747/video'):
                return True

        # 3. Ưu tiên 2: Thử RealSense D435 nếu có cắm
        if self.video_device == 'auto' or self.video_device == 'd435':
            if self._open_realsense():
                return True
            if self.video_device == 'd435':
                return False

        # 4. Tự động dò iPhone / V4L2 device
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
                        f"⏳ Đang tìm camera (RealSense D435 hoặc iPhone qua Iriun/USB)...",
                        throttle_duration_sec=3.0
                    )
                    time.sleep(1.5)
                    continue
                consecutive_fail = 0

            frame = None

            # 1. Đọc từ RealSense D435
            if self._mode == 'realsense':
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        raise RuntimeError("No color frame from RealSense")
                    frame = np.asanyarray(color_frame.get_data())
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

            # 2. Đọc từ OpenCV (iPhone / USB Webcam)
            elif self._mode == 'opencv':
                try:
                    ret, raw_frame = self._cap.read()
                    if not ret or raw_frame is None or raw_frame.size == 0:
                        raise RuntimeError("Failed to grab frame from OpenCV")

                    # Resize nếu kích thước nhận được khác với cấu hình
                    fh, fw = raw_frame.shape[:2]
                    if fw != self.width or fh != self.height:
                        frame = cv2.resize(raw_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame = raw_frame

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

            if frame is None:
                continue

            # Chuẩn bị ROS 2 Image Message
            ph, pw = frame.shape[:2]
            msg = Image()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.height          = ph
            msg.width           = pw
            msg.encoding        = 'bgr8'
            msg.is_bigendian    = False
            msg.step            = pw * 3
            msg.data            = frame.tobytes()

            if not self.is_running or not rclpy.ok():
                break

            try:
                self.image_pub.publish(msg)
                self.image_alt_pub.publish(msg)
            except Exception:
                break

            # Log thống kê mỗi 15 giây
            self._frame_count += 1
            now = time.time()
            if now - self._last_log >= 15.0:
                fps_actual = self._frame_count / (now - self._last_log)
                self.get_logger().info(
                    f"📷 [{self._source_name}] {pw}x{ph} | {fps_actual:.1f} FPS | "
                    f"{self._frame_count} frames → AI CNN")
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
