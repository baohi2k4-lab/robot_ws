#!/usr/bin/env python3
"""
==============================================================================
  Script: train_yolo_arrows.py
  Mục đích:
    Huấn luyện model YOLOv8 siêu nhẹ (YOLOv8n) nhận diện 2 lớp:
      - 0: yellow_arrow (Mũi tên vàng - Điều kiện 1)
      - 1: black_arrow  (Mũi tên đen - Điều kiện 2)
    Sau khi train xong, script sẽ tự động xuất ra file 'yolo_arrow.onnx'
    và lưu vào thư mục '~/robot_ws/models/' để robot sử dụng trực tiếp.

  Cách chạy:
    1. Cài đặt thư viện huấn luyện (nếu chưa có):
       pip install ultralytics

    2. Chạy huấn luyện:
       python3 scripts/train_yolo_arrows.py
==============================================================================
"""

import os
import sys

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Chưa cài đặt thư viện 'ultralytics'. Vui lòng chạy lệnh sau:")
        print("   pip install ultralytics")
        sys.exit(1)

    ws_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(ws_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    yaml_path = os.path.join(ws_dir, 'dataset_arrows', 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"⚠️ Chưa tìm thấy file cấu hình dataset: {yaml_path}")
        print("📝 Đang tạo file cấu hình mẫu data.yaml...")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            f.write(f"""path: {os.path.join(ws_dir, 'dataset_arrows')}
train: images/train
val: images/val

names:
  0: yellow_arrow
  1: black_arrow
""")
        print(f"✅ Đã tạo {yaml_path}")

    print("============================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN YOLOv8n (MŨI TÊN VÀNG & ĐEN)")
    print("============================================================")

    # Khởi tạo model YOLOv8n nano siêu nhẹ
    model = YOLO('yolov8n.pt')

    # Huấn luyện 50 - 100 epochs
    results = model.train(
        data=yaml_path,
        epochs=60,
        imgsz=640,
        batch=16,
        name='arrow_yolo_model'
    )

    print("\n📦 Đang xuất model sang định dạng ONNX tăng tốc cho Robot...")
    onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
    
    target_onnx = os.path.join(models_dir, 'yolo_arrow.onnx')
    import shutil
    shutil.copyfile(onnx_path, target_onnx)
    print(f"🎉 HUẤN LUYỆN & XUẤT MODEL THÀNH CÔNG!")
    print(f"   Đã lưu model vào: {target_onnx}")
    print("   Robot sẽ tự động nhận diện và sử dụng model này!")

if __name__ == '__main__':
    main()
