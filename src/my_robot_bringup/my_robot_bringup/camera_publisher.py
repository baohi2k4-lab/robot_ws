#!/usr/bin/env python3
"""
Camera Publisher Node — Intel RealSense D435 (Color Stream RGB).

Chỉ hỗ trợ Intel RealSense D435 qua pyrealsense2.
Publish ảnh màu BGR8 640×480 lên /camera/color/image_raw.

Sử dụng:
  ros2 run my_robot_bringup camera_publisher
  ros2 run my_robot_bringup camera_publisher --ros-args -p width:=1280 -p height:=720 -p fps:=30.0
"""

import time
import threading

import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    """Publish ảnh màu từ Intel RealSense D435 qua pyrealsense2."""

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('camera_frame_id', 'camera_link')

        self.width  = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps    = float(self.get_parameter('fps').value)
        self.frame_id = self.get_parameter('camera_frame_id').value

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
        self.is_running   = True
        self._pipeline    = None
        self._frame_count = 0
        self._last_log    = time.time()

        # ── Start thread ──────────────────────────────────────────────
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ── RealSense helpers ──────────────────────────────────────────────
    def _open_realsense(self) -> bool:
        """Mở RealSense D435 Color stream. Trả về True nếu thành công."""
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

            # Đọc thử 1 frame
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            if not frames.get_color_frame():
                pipeline.stop()
                return False

            self._pipeline = pipeline
            self.get_logger().info(
                f"📷 [D435 READY] {name} | S/N: {serial} | "
                f"{self.width}x{self.height} @ {int(self.fps)} FPS | Color BGR8")
            return True

        except Exception as e:
            self.get_logger().warn(
                f"⏳ Chờ RealSense D435... ({e})", throttle_duration_sec=3.0)
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

    # ── Main loop ──────────────────────────────────────────────────────
    def _capture_loop(self):
        consecutive_fail = 0

        while self.is_running:
            # Kết nối nếu chưa có pipeline
            if self._pipeline is None:
                if not self._open_realsense():
                    time.sleep(1.5)
                    continue
                consecutive_fail = 0

            # Đọc frame
            try:
                frames      = self._pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("No color frame")
                frame = np.asanyarray(color_frame.get_data())
                consecutive_fail = 0
            except Exception:
                consecutive_fail += 1
                if consecutive_fail >= 30:   # ~3s không có frame → reconnect
                    self.get_logger().warn(
                        "Mất tín hiệu D435, đang kết nối lại...",
                        throttle_duration_sec=5.0)
                    self._close_realsense()
                time.sleep(0.1)
                continue

            # Publish
            ph, pw = frame.shape[:2]
            msg = Image()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.height   = ph
            msg.width    = pw
            msg.encoding = 'bgr8'
            msg.is_bigendian = False
            msg.step = pw * 3
            msg.data = frame.tobytes()

            if not self.is_running or not rclpy.ok():
                break
            try:
                self.image_pub.publish(msg)
                self.image_alt_pub.publish(msg)
            except Exception:
                break

            # Log mỗi 30s
            self._frame_count += 1
            now = time.time()
            if now - self._last_log >= 30.0:
                fps_actual = self._frame_count / (now - self._last_log)
                self.get_logger().info(
                    f"📷 [D435] {pw}x{ph} | {fps_actual:.1f} FPS | "
                    f"{self._frame_count} frames → AI CNN")
                self._frame_count = 0
                self._last_log = now

    def destroy_node(self):
        self.is_running = False
        self._close_realsense()
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
