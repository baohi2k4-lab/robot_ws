#!/usr/bin/env python3
"""
Giao diện trực quan xem Camera RealSense D435 & AI CNN thời gian thực:
- Luồng 1: Ảnh màu RGB 640x480 @ 30 FPS kèm HUD mạng AI CNN tìm tim luống và góc bẻ lái.
- Luồng 2: Bản đồ nhiệt độ sâu Depth Map 16UC1 quang học (IR Stereo mm, đo khoảng cách mọi điểm ảnh).
- Hiển thị tâm ngắm, FPS thực tế, cự ly vật thể tại tâm (cm/m) và hỗ trợ rê chuột đo khoảng cách.
- Phím tắt:
    'q' hoặc 'ESC': Thoát
    's': Lưu ảnh chụp màn hình kiểm chứng vào thư mục images/
    'c': Bật / Tắt lớp vẽ AI CNN
"""

import os
import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    print("❌ Chưa cài đặt OpenCV (cv2)!")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WS_DIR, 'src', 'my_robot_controller'))

try:
    from my_robot_controller.inference_handler import InferenceHandler
except ImportError:
    InferenceHandler = None

# Tọa độ chuột để đo khoảng cách theo điểm trỏ
mouse_x, mouse_y = 320, 240

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        # Nếu chuột nằm ở nửa bên phải (Depth Map: x >= 640)
        if x >= 640:
            mouse_x, mouse_y = x - 640, y
        else:
            mouse_x, mouse_y = x, y

def main():
    print("=" * 75)
    print("📷 [XEM CAMERA LIVE] INTEL REALSENSE D435 & AI CNN THEO DÕI THỜI GIAN THỰC")
    print("   • Nhấn 'q' hoặc 'ESC' trên cửa sổ để THOÁT.")
    print("   • Nhấn 's' để LƯU ẢNH chụp màn hình.")
    print("   • Nhấn 'c' để BẬT/TẮT lớp phủ AI CNN.")
    print("   • Rê chuột trên ảnh để ĐO KHOẢNG CÁCH tại bất kỳ điểm nào!")
    print("=" * 75)

    # 1. Khởi tạo Model AI CNN
    model_path = os.path.join(WS_DIR, 'src', 'my_robot_controller', 'models', 'crop_row_cnn_best_final_int8.onnx')
    infer = None
    if InferenceHandler and os.path.exists(model_path):
        try:
            infer = InferenceHandler(model_path=model_path, input_size=(384, 384), mask_threshold=0.30, use_hsv_mask=False)
            print(f"✅ Đã tải Model AI CNN: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"⚠️ Không thể tải model ONNX: {e}")

    # 2. Mở Camera (Ưu tiên RealSense D435)
    pipe = None
    align = None
    cap = None
    is_realsense = False

    if rs is not None:
        try:
            ctx = rs.context()
            if len(ctx.query_devices()) > 0:
                dev = ctx.query_devices()[0]
                dev_name = dev.get_info(rs.camera_info.name)
                dev_sn = dev.get_info(rs.camera_info.serial_number)
                usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                print(f"🎯 Đang mở camera phần cứng: {dev_name} (S/N: {dev_sn} | USB {usb_type})")

                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipe.start(cfg)
                align = rs.align(rs.stream.color)
                is_realsense = True
                print("✅ RealSense D435: Đã kích hoạt song song 2 luồng RGB 640x480 + Depth Map 16UC1 @ 30 FPS!")
        except Exception as e:
            print(f"⚠️ Lỗi mở RealSense: {e}. Thử fallback webcam...")
            pipe = None

    if not is_realsense:
        # Fallback OpenCV Webcam / DroidCam
        for dev_target in ['http://172.20.10.1:4747/video', 0, 1, 2]:
            c = cv2.VideoCapture(dev_target)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    cap = c
                    print(f"📷 Đã mở Webcam/Phone tại nguồn: {dev_target}")
                    break
                c.release()

    if pipe is None and cap is None:
        print("❌ LỖI: Không tìm thấy camera nào khả dụng trên máy!")
        return

    win_name = "Camera RealSense D435 Live (RGB + Depth + AI CNN)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 520)
    cv2.setMouseCallback(win_name, on_mouse)

    show_ai = True
    frame_count = 0
    fps_display = 30.0
    t_last_fps = time.time()

    try:
        while True:
            t_frame_start = time.time()
            color_img = None
            depth_img = None

            if is_realsense:
                try:
                    frames = pipe.wait_for_frames(1000)
                    if align:
                        frames = align.process(frames)
                    cf = frames.get_color_frame()
                    df = frames.get_depth_frame()
                    if cf:
                        color_img = np.asanyarray(cf.get_data())
                    if df:
                        depth_img = np.asanyarray(df.get_data())
                    del frames, cf, df
                except Exception:
                    continue
            else:
                ret, raw = cap.read()
                if ret and raw is not None:
                    color_img = cv2.resize(raw, (640, 480))

            if color_img is None:
                continue

            h, w = color_img.shape[:2]
            rgb_hud = color_img.copy()

            # 3. Chạy suy luận AI CNN
            steer_angle = 0.0
            confidence = 0.0
            infer_ms = 0.0

            if infer and show_ai:
                t0 = time.time()
                steer_angle, offset, lane_center, confidence = infer.process_image(color_img, max_angle_deg=16.0)
                infer_ms = (time.time() - t0) * 1000.0

                img_cx = int((w - 1) / 2.0)
                lane_cx = int(np.clip(lane_center * (w / 384.0), 0, w - 1))

                # Vẽ vạch tim xe (màu xanh lá)
                for y_s in range(0, h, 25):
                    cv2.line(rgb_hud, (img_cx, y_s), (img_cx, min(y_s + 15, h)), (0, 255, 0), 2)

                # Vẽ vạch tim luống AI bẻ lái (màu cam)
                cv2.line(rgb_hud, (lane_cx, h - 1), (lane_cx, int(h * 0.35)), (0, 140, 255), 3)
                cv2.circle(rgb_hud, (lane_cx, int(h * 0.35)), 7, (0, 140, 255), -1)

            # Vẽ thông tin lên khung RGB
            cv2.putText(rgb_hud, "1. RGB Camera (640x480)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(rgb_hud, f"FPS: {fps_display:.1f} | AI: {infer_ms:.1f}ms", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if show_ai:
                conf_color = (0, 255, 0) if confidence >= 0.30 else (0, 165, 255)
                cv2.putText(rgb_hud, f"Conf: {confidence*100:.1f}% | Steer: {steer_angle:+.2f} deg", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2)

            # 4. Xử lý bản đồ nhiệt Depth Map
            if depth_img is not None:
                # Chuyển thang đo 0-4m (0-4000 mm) sang colormap JET
                depth_scaled = cv2.convertScaleAbs(depth_img, alpha=255.0 / 4000.0)
                depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
                depth_colormap[depth_img == 0] = [0, 0, 0]  # Điểm mù hiển thị đen

                # Đo khoảng cách tại tâm và tại vị trí chuột
                cy, cx = h // 2, w // 2
                center_d_mm = depth_img[cy, cx]

                # Đảm bảo tọa độ chuột hợp lệ
                mx = max(0, min(w - 1, mouse_x))
                my = max(0, min(h - 1, mouse_y))
                mouse_d_mm = depth_img[my, mx]

                # Vẽ tâm ngắm chữ thập
                cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(rgb_hud, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # Vẽ điểm đo theo chuột
                cv2.circle(depth_colormap, (mx, my), 5, (255, 255, 0), 2)

                cv2.putText(depth_colormap, "2. Hardware Depth Map (IR Stereo)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(depth_colormap, f"Tam ngam: {center_d_mm} mm ({center_d_mm/1000.0:.2f} m)", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(depth_colormap, f"Diem tro: {mouse_d_mm} mm ({mouse_d_mm/1000.0:.2f} m) tai ({mx},{my})", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            else:
                # Nếu không có luồng depth, tạo màn hình đen thông báo
                depth_colormap = np.zeros_like(color_img)
                cv2.putText(depth_colormap, "Depth Map: Chi ho tro tren RealSense D435", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Ghép đôi song song 2 màn hình (Ngang: 1280, Dọc: 480)
            combined_view = np.hstack((rgb_hud, depth_colormap))

            # Thanh trạng thái dưới đáy
            status_bar = np.zeros((35, combined_view.shape[1], 3), dtype=np.uint8)
            status_bar[:] = (30, 30, 30)
            status_text = "[Phim tat]  'q' / ESC: Thoat  |  's': Chup luu anh  |  'c': An/Hien AI CNN  |  Re chuot de do khoang cach mm"
            cv2.putText(status_bar, status_text, (20, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            final_screen = np.vstack((combined_view, status_bar))
            cv2.imshow(win_name, final_screen)

            # Tính toán FPS
            frame_count += 1
            now_t = time.time()
            if now_t - t_last_fps >= 1.0:
                fps_display = frame_count / (now_t - t_last_fps)
                frame_count = 0
                t_last_fps = now_t

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                out_path = os.path.join(WS_DIR, 'images', f'snapshot_{int(time.time())}.jpg')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, final_screen)
                print(f"📸 Đã lưu ảnh chụp màn hình tại: {out_path}")
            elif key == ord('c'):
                show_ai = not show_ai

    except KeyboardInterrupt:
        pass
    finally:
        if pipe:
            pipe.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Đã đóng cửa sổ camera.")

if __name__ == '__main__':
    main()
