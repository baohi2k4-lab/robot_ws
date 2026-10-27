#!/usr/bin/env python3
"""
==============================================================================
  📸 CÔNG CỤ THU THẬP DỮ LIỆU HUẤN LUYỆN YOLO (MŨI TÊN VÀ VẬT THỂ)
  Thiết bị: Intel RealSense D435 (RGB + Depth Stream)
  Thư mục lưu: ~/Downloads/anh_yolo/
==============================================================================
  Phím bấm điều khiển:
    [SPACE] hoặc [c] : Chụp 1 ảnh (Tên: yolo_XXXX.jpg)
    [y]              : Chụp Mũi tên VÀNG (Tên: yolo_vang_XXXX.jpg)
    [b]              : Chụp Mũi tên ĐEN (Tên: yolo_den_XXXX.jpg)
    [a]              : BẬT / TẮT TỰ ĐỘNG CHỤP (Mỗi 1.5s chụp 1 ảnh khi quay/đổi góc)
    [q] hoặc [ESC]   : Lưu và THOÁT
==============================================================================
"""

import os
import sys
import time
import subprocess
import cv2
import numpy as np

# Thư mục đích lưu ảnh theo yêu cầu: ~/Downloads/anh_yolo/
SAVE_DIR = os.path.join(os.path.expanduser('~'), 'Downloads', 'anh_yolo')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'vang'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'den'), exist_ok=True)


def check_and_kill_busy_camera():
    """Tự động kiểm tra và giải phóng camera nếu có tiến trình khác đang chiếm giữ."""
    try:
        out = subprocess.check_output("fuser /dev/video* 2>/dev/null || true", shell=True).decode()
        pids = [p.strip() for p in out.split() if p.strip().isdigit()]
        curr_pid = str(os.getpid())
        for pid in pids:
            if pid != curr_pid:
                # Kiểm tra nếu là view_camera_live thì dừng để lấy quyền camera
                cmd = subprocess.check_output(f"ps -p {pid} -o comm= 2>/dev/null || true", shell=True).decode().strip()
                if 'python' in cmd.lower():
                    print(f"🔄 Đang giải phóng camera từ tiến trình cũ (PID: {pid})...")
                    subprocess.run(f"kill -9 {pid} 2>/dev/null || true", shell=True)
                    time.sleep(0.5)
    except Exception:
        pass


def count_existing_images(directory, prefix=""):
    """Đếm số ảnh đã có để đánh số thứ tự tiếp tục không bị trùng."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for f in os.listdir(directory):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith(prefix):
            count += 1
    return count


def main():
    print("=" * 70)
    print("🚀 [CHỤP ẢNH TRAIN YOLO] INTEL REALSENSE D435")
    print(f"📁 Thư mục lưu ảnh: {SAVE_DIR}")
    print("=" * 70)

    check_and_kill_busy_camera()

    pipeline = None
    align = None
    cap = None

    # Khởi tạo RealSense Pipeline
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            has_depth = True
        except Exception:
            has_depth = False

        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("✅ Đã kết nối thành công Camera Intel RealSense D435!")
    except Exception as e:
        print(f"⚠️ Không mở được RealSense API ({e}). Đang thử mở qua OpenCV VideoCapture...")
        for dev_id in [3, 4, 5, 2, 0]:
            cap = cv2.VideoCapture(dev_id)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"✅ Mở thành công Camera qua OpenCV /dev/video{dev_id}")
                    break
                cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            print("❌ Không tìm thấy luồng camera nào khả dụng! Hãy kiểm tra cáp USB RealSense.")
            sys.exit(1)

    # Đếm số ảnh ban đầu để tiếp tục đánh số
    idx_all = count_existing_images(SAVE_DIR, "yolo_")
    idx_vang = count_existing_images(os.path.join(SAVE_DIR, 'vang'), "yolo_vang_")
    idx_den = count_existing_images(os.path.join(SAVE_DIR, 'den'), "yolo_den_")

    auto_capture = False
    last_auto_time = 0.0
    auto_interval = 1.5  # giây giữa các lần tự động chụp

    flash_text = ""
    flash_time = 0.0

    win_name = "YOLO Dataset Collector - RealSense D435 (~/Downloads/anh_yolo)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            dist_str = "N/A"
            center_dist_mm = 0

            if pipeline is not None:
                try:
                    frames = pipeline.wait_for_frames(1500)
                    aligned = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                    if depth_frame:
                        depth = np.asanyarray(depth_frame.get_data())
                        center_dist_mm = depth[240, 320]
                        if center_dist_mm > 0:
                            dist_str = f"{center_dist_mm/1000.0:.2f}m"
                except Exception as e:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            h, w = frame.shape[:2]
            display = frame.copy()

            # Chế độ tự động chụp
            now = time.time()
            if auto_capture and (now - last_auto_time >= auto_interval):
                idx_all += 1
                fname = f"yolo_{idx_all:04d}.jpg"
                save_path = os.path.join(SAVE_DIR, fname)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_auto_time = now
                flash_text = f"AUTO: {fname}"
                flash_time = now
                print(f"🤖 [AUTO] Đã lưu: {fname}")

            # Vẽ HUD thông tin
            cv2.rectangle(display, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), -1)

            # Dòng 1: Thống kê số lượng
            cv2.putText(display, f"YOLO: {idx_all} | Vang: {idx_vang} | Den: {idx_den} | Khoang cach: {dist_str}",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Dòng 2: Hướng dẫn phím bấm
            auto_tag = "[DANG BAT]" if auto_capture else "[TAT]"
            auto_color = (0, 255, 0) if auto_capture else (180, 180, 180)
            cv2.putText(display, f"[SPACE]: Chup | [y]: Vang | [b]: Den | [a]: Auto {auto_tag} | [q]: Thoat",
                        (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            # Dòng đáy: Thư mục lưu
            cv2.putText(display, f"Thu muc: ~/Downloads/anh_yolo/",
                        (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            # Tâm ngắm chữ thập
            cx, cy = w // 2, h // 2
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # Thông báo vừa chụp xong (hiện trong 1s)
            if now - flash_time < 1.0 and flash_text:
                cv2.putText(display, f"DA LUU: {flash_text}", (w // 2 - 140, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in [32, ord('c')]:  # SPACE hoặc 'c' -> Chụp ảnh chung
                idx_all += 1
                fname = f"yolo_{idx_all:04d}.jpg"
                save_path = os.path.join(SAVE_DIR, fname)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                flash_text = fname
                flash_time = now
                print(f"📸 [+{idx_all}] Đã lưu ảnh: {fname}")

            elif key == ord('y'):  # 'y' -> Chụp mũi tên Vàng
                idx_vang += 1
                fname = f"yolo_vang_{idx_vang:04d}.jpg"
                save_path = os.path.join(SAVE_DIR, 'vang', fname)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Lưu đồng thời bản sao vào all để tiện train
                cv2.imwrite(os.path.join(SAVE_DIR, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                flash_text = fname
                flash_time = now
                print(f"🟡 [+{idx_vang}] Đã lưu Mũi tên VÀNG: {fname} (Depth: {dist_str})")

            elif key == ord('b'):  # 'b' -> Chụp mũi tên Đen
                idx_den += 1
                fname = f"yolo_den_{idx_den:04d}.jpg"
                save_path = os.path.join(SAVE_DIR, 'den', fname)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Lưu đồng thời bản sao vào all để tiện train
                cv2.imwrite(os.path.join(SAVE_DIR, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                flash_text = fname
                flash_time = now
                print(f"⚫ [+{idx_den}] Đã lưu Mũi tên ĐEN: {fname} (Depth: {dist_str})")

            elif key == ord('a'):  # Bật/tắt chế độ tự động chụp
                auto_capture = not auto_capture
                last_auto_time = now
                state_str = "BẬT" if auto_capture else "TẮT"
                print(f"⏱️ Chế độ tự động chụp mỗi {auto_interval}s: {state_str}")

            elif key in [ord('q'), 27]:  # Thoát
                break

    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("🎉 [HOÀN TẤT THU THẬP DATASET YOLO]")
        print(f"📁 Toàn bộ ảnh đã được lưu tại: {SAVE_DIR}")
        print(f"   • Tổng số ảnh yolo: {idx_all}")
        print(f"   • Mũi tên vàng    : {idx_vang}")
        print(f"   • Mũi tên đen     : {idx_den}")
        print("=" * 70)


if __name__ == '__main__':
    main()
