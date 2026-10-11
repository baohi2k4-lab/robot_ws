#!/usr/bin/env python3
"""
Kiểm tra chẩn đoán trực tiếp: Camera -> ROS 2 Image Topic -> AI CNN ONNX.
"""
import os, sys, time, cv2, numpy as np

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_controller'))

from my_robot_controller.inference_handler import InferenceHandler

def main():
    print("=" * 70)
    print("🔍 KIỂM TRA CHUỖI NHẬN ẢNH VÀ XỬ LÝ CNN TRÊN LAPTOP")
    print("=" * 70)

    model_path = os.path.join(WS_DIR, 'src', 'my_robot_controller', 'models', 'crop_row_cnn_best_final_int8.onnx')
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        return

    infer = InferenceHandler(model_path=model_path, input_size=(384, 384), mask_threshold=0.25, use_hsv_mask=False)
    print(f"✅ Đã load Model ONNX: {os.path.basename(model_path)}")

    # 1. Thử nhận frame từ ROS 2 topic nếu node đang chạy
    import rclpy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()
    node = rclpy.create_node('camera_verifier')
    bridge = CvBridge()
    captured_frame = None

    def cb(msg):
        nonlocal captured_frame
        try:
            captured_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(f"Lỗi cv_bridge: {e}")

    sub = node.create_subscription(Image, '/camera/color/image_raw', cb, 10)
    print("⏳ Đang lắng nghe 3s trên topic /camera/color/image_raw xem node phát ảnh có đang gửi không...")
    start_t = time.time()
    while time.time() - start_t < 3.0 and captured_frame is None:
        rclpy.spin_once(node, timeout_sec=0.2)

    sub.destroy()
    node.destroy_node()
    rclpy.shutdown()

    if captured_frame is not None:
        print(f"✅ THÀNH CÔNG: Đã nhận được ảnh TRỰC TIẾP từ topic /camera/color/image_raw!")
        print(f"   • Kích thước frame nhận được: {captured_frame.shape[1]}x{captured_frame.shape[0]} (3 kênh BGR)")
    else:
        print("ℹ️ Chưa thấy topic /camera/color/image_raw (chưa bật laptop-ai).")
        # Thử chụp trực tiếp từ RealSense hoặc DroidCam
        print("⏳ Đang thử mở camera trực tiếp (RealSense D435 hoặc DroidCam)...")
        cap = None
        # Thử RealSense
        try:
            import pyrealsense2 as rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipe.start(cfg)
            for _ in range(10): pipe.wait_for_frames(1000)
            f = pipe.wait_for_frames(1000)
            cf = f.get_color_frame()
            captured_frame = np.asanyarray(cf.get_data())
            pipe.stop()
            print("✅ Đã chụp thành công 1 frame từ RealSense D435!")
        except Exception:
            pass

        if captured_frame is None:
            # Thử DroidCam
            for url in ['http://172.20.10.1:4747/video', 'http://127.0.0.1:4747/video']:
                c = cv2.VideoCapture(url)
                if c.isOpened():
                    ret, f = c.read()
                    if ret and f is not None:
                        captured_frame = f
                        print(f"✅ Đã chụp thành công 1 frame từ DroidCam ({url})!")
                    c.release()
                if captured_frame is not None:
                    break

        if captured_frame is None:
            for dev_idx in [0, 1, 2]:
                c = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                if c.isOpened():
                    ret, f = c.read()
                    if ret and f is not None:
                        captured_frame = f
                        print(f"✅ Đã chụp thành công 1 frame từ /dev/video{dev_idx}!")
                    c.release()
                if captured_frame is not None:
                    break

    if captured_frame is None:
        print("❌ Chưa chụp được frame từ camera nào (hãy mở app DroidCam hoặc cắm camera).")
        return

    # 2. Chạy qua mạng CNN
    t0 = time.time()
    heading, offset, center, conf = infer.process_image(captured_frame, max_angle_deg=15.0)
    dur_ms = (time.time() - t0) * 1000.0

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ MẠNG AI CNN SUY LUẬN TRÊN ẢNH CAMERA:")
    print("=" * 70)
    print(f"   • Kích thước ảnh đầu vào: {captured_frame.shape[1]}x{captured_frame.shape[0]}")
    print(f"   • Thời gian xử lý AI: {dur_ms:.1f} ms (~{1000.0/max(dur_ms, 1.0):.1f} FPS)")
    print(f"   • Độ tin cậy (Confidence): {conf * 100:.1f}%")
    print(f"   • Góc lái tính được: {heading:+.2f}°")
    print(f"   • Lệch tim luống (Offset): {offset:+.2f}")

    # Vẽ HUD và lưu ảnh
    h, w = captured_frame.shape[:2]
    img_cx = int((w - 1) / 2.0)
    lane_cx = int(np.clip(center * (w / 384.0), 0, w - 1))
    hud = captured_frame.copy()
    cv2.line(hud, (img_cx, 0), (img_cx, h), (0, 255, 0), 2)
    cv2.line(hud, (lane_cx, h - 1), (lane_cx, int(h * 0.4)), (255, 140, 0), 3)
    cv2.putText(hud, f"Conf: {conf*100:.1f}% | Steer: {heading:+.2f} deg", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    out_path = os.path.join(WS_DIR, 'images', 'kiem_tra_cnn_hien_tai.jpg')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, hud)
    print(f"   • File ảnh HUD kiểm chứng đã lưu tại: {out_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
