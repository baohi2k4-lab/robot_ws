#!/usr/bin/env python3
"""
CNN Inference Server (Chạy trên Laptop để gánh toàn bộ tác vụ AI cho Pi).
Chức năng:
  - Nhận luồng ảnh nén (/camera/compressed) hoặc raw (/camera/color/image_raw) từ Pi qua Wi-Fi/LAN.
  - Chạy mô hình ONNX trên Laptop với CPU/GPU mạnh (tốc độ 30 - 60+ FPS, độ trễ ~10-15ms).
  - Trích xuất: góc lệch hướng (heading_error), độ lệch tim luống (lane_offset), tọa độ tâm (lane_center), độ tin cậy (confidence).
  - Bắn kết quả về Pi qua topic '/crop_row/detection' (Float32MultiArray).
  - Node 'cnn_driver' trên Pi nhận kết quả này để ra quyết định điều khiển (dừng xoay IMU, né cản, bẻ lái).

Sử dụng trên Laptop:
  ros2 run my_robot_controller cnn_server
  (hoặc dùng lệnh tắt: laptop-cnn)
"""

import os
import sys
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray

from .inference_handler import InferenceHandler


class CnnInferenceServer(Node):
    def __init__(self):
        super().__init__('cnn_inference_server')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('model_path', 'models/crop_row_cnn_best.onnx')
        self.declare_parameter('input_height', 384)
        self.declare_parameter('input_width', 384)
        self.declare_parameter('mask_threshold', 0.35)
        self.declare_parameter('max_steering_angle_deg', 16.0)
        self.declare_parameter('num_threads', 0)
        self.declare_parameter('show_window', False)

        p = self.get_parameter
        self.model_path = p('model_path').value
        self.input_height = p('input_height').value
        self.input_width = p('input_width').value
        self.mask_threshold = p('mask_threshold').value
        self.max_steering_angle_deg = p('max_steering_angle_deg').value
        self.num_threads = p('num_threads').value
        self.show_window = p('show_window').value

        # Tự động tìm model ONNX nếu đường dẫn tương đối
        if not os.path.exists(self.model_path):
            candidate_names = [os.path.basename(self.model_path), 'crop_row_cnn_best.onnx', 'crop_row_cnn_best_final.onnx', 'crop_row_cnn_best_final_int8.onnx']
            try:
                from ament_index_python.packages import get_package_share_directory
                share_dir = get_package_share_directory('my_robot_controller')
                for name in candidate_names:
                    cand = os.path.join(share_dir, 'models', name)
                    if os.path.exists(cand):
                        self.model_path = cand
                        break
            except Exception:
                pass

        if not os.path.exists(self.model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            for name in candidate_names:
                cand = os.path.abspath(os.path.join(current_dir, '..', 'models', name))
                if os.path.exists(cand):
                    self.model_path = cand
                    break

        self.get_logger().info(f"💻 [CNN Server] Nạp mô hình ONNX: {self.model_path}")
        self.inference = InferenceHandler(
            model_path=self.model_path,
            mask_threshold=self.mask_threshold,
            input_size=(self.input_height, self.input_width),
            use_hsv_mask=False,
            num_threads=self.num_threads
        )

        # ── Publishers ────────────────────────────────────────────────
        # Topic bắn kết quả phát hiện về Pi (gọn nhẹ ~20 bytes, truyền cực nhanh qua Wi-Fi)
        qos_det = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.detection_pub = self.create_publisher(Float32MultiArray, '/crop_row/detection', qos_det)
        self.debug_img_pub = self.create_publisher(CompressedImage, '/crop_row/debug_mask/compressed', qos_det)

        # ── Subscribers ───────────────────────────────────────────────
        # 1. Nhận ảnh nén JPEG từ Pi (Ưu tiên: siêu nhẹ trên Wi-Fi ~20KB/frame)
        qos_compressed = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.compressed_sub = self.create_subscription(
            CompressedImage, '/camera/compressed', self.compressed_image_callback, qos_compressed
        )

        # 2. Nhận ảnh raw (Nếu nối cáp LAN tốc độ cao hoặc local)
        qos_raw = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.raw_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.raw_image_callback, qos_raw
        )

        # ── Metrics ───────────────────────────────────────────────────
        self._last_process_time = time.time()
        self._frame_count = 0
        self._fps_timer = self.create_timer(1.0, self._log_fps)
        self._latest_fps = 0.0
        self._latest_latency_ms = 0.0

        self.get_logger().info("🚀 [CNN Server Laptop] Sẵn sàng xử lý AI! Đang lắng nghe luồng ảnh từ Pi...")

    def compressed_image_callback(self, msg: CompressedImage):
        """Giải nén JPEG và chạy inference."""
        t_start = time.time()
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                return
            self._process_and_publish(bgr_img, t_start)
        except Exception as e:
            self.get_logger().warn(f"Lỗi giải nén: {e}", throttle_duration_sec=2.0)

    def raw_image_callback(self, msg: Image):
        """Xử lý trực tiếp ảnh raw."""
        # Nếu đã có luồng compressed đang chạy thì bỏ qua raw để tránh xử lý trùng
        now = time.time()
        if (now - getattr(self, '_last_compressed_time', 0.0)) < 0.2:
            return

        t_start = time.time()
        try:
            channels = 3
            expected_size = msg.height * msg.width * channels
            if len(msg.data) < expected_size:
                return
            img = np.frombuffer(msg.data, dtype=np.uint8)[:expected_size].reshape(msg.height, msg.width, channels)
            if msg.encoding in ('rgb8', 'RGB8'):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self._process_and_publish(img, t_start)
        except Exception as e:
            self.get_logger().warn(f"Lỗi đọc raw: {e}", throttle_duration_sec=2.0)

    def _process_and_publish(self, bgr_img: np.ndarray, t_start: float):
        self._last_compressed_time = time.time()
        heading_error, lane_offset, lane_center, confidence = self.inference.process_image(
            bgr_img, self.max_steering_angle_deg
        )
        latency_ms = (time.time() - t_start) * 1000.0
        self._latest_latency_ms = latency_ms

        # Bắn kết quả về Pi
        out_msg = Float32MultiArray()
        # [0]: heading_error (độ)
        # [1]: lane_offset (-1.0 đến 1.0)
        # [2]: lane_center (pixel X)
        # [3]: confidence (0.0 đến 1.0)
        # [4]: latency_ms (thời gian tính toán)
        out_msg.data = [
            float(heading_error),
            float(lane_offset),
            float(lane_center),
            float(confidence),
            float(latency_ms)
        ]
        self.detection_pub.publish(out_msg)
        self._frame_count += 1

        # Hiển thị cửa sổ debug nếu bật show_window
        if self.show_window:
            h, w = bgr_img.shape[:2]
            cv2.putText(
                bgr_img,
                f"Angle: {heading_error:+.1f} deg | Conf: {confidence:.2f} | {latency_ms:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("CNN Server (Laptop)", bgr_img)
            cv2.waitKey(1)

    def _log_fps(self):
        fps = self._frame_count
        self._frame_count = 0
        self._latest_fps = fps
        if fps > 0:
            self.get_logger().info(
                f"⚡ [Laptop CNN Worker] Tốc độ: {fps} FPS | Độ trễ suy luận: {self._latest_latency_ms:.1f} ms"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CnnInferenceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
