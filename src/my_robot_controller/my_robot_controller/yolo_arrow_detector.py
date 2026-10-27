#!/usr/bin/env python3
"""
==============================================================================
  Module: yolo_arrow_detector.py
  Package: my_robot_controller
  
  Mục đích:
    Nhận diện Mũi tên Vàng (Điều kiện 1) và Mũi tên Đen (Điều kiện 2)
    sử dụng mô hình YOLO (.onnx) chạy trực tiếp trên onnxruntime.
    Kết hợp tọa độ Bounding Box với bản đồ chiều sâu RealSense Depth Map
    để tính toán cự ly thực tế (mét) và độ lệch tâm (offset) của mũi xe.

  Classes nhận diện:
    0: yellow_arrow (Mũi tên vàng - Điểm dừng 0.5m để quay)
    1: black_arrow  (Mũi tên đen - Miệng luống 2 để căn tâm vào luống)
==============================================================================
"""

import os
import time
import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class YoloArrowDetector:
    def __init__(self, model_path=None, conf_threshold=0.30, iou_threshold=0.45,
                 class_conf_thresholds=None, confirm_frames=3, confirm_reset_sec=1.0):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self.input_shape = (640, 640)
        self.classes = ['yellow_arrow', 'black_arrow']

        # Ngưỡng tin cậy riêng từng lớp (tối ưu cho model bao1.onnx đã lọc sạch nhãn nền).
        # Mũi tên đen khi vừa mới lọt vào mép phải ở xa (~1.5m, frame 0070) có conf ~0.33, nền = 0.001.
        self.class_conf_thresholds = {'yellow_arrow': 0.55, 'black_arrow': 0.32}
        if class_conf_thresholds:
            self.class_conf_thresholds.update(class_conf_thresholds)

        # Xác nhận theo thời gian: chỉ báo detected=True khi thấy liên tiếp N lần gọi detect().
        # Nếu giữa 2 lần gọi cách nhau > confirm_reset_sec (YOLO bị tắt theo FSM) thì đếm lại từ đầu.
        self.confirm_frames = max(1, int(confirm_frames))
        self.confirm_reset_sec = float(confirm_reset_sec)
        self._hit_streak = {c: 0 for c in self.classes}
        self._last_detect_time = None

        # Tìm model ONNX
        resolved_path = self._resolve_model_path(model_path)
        if resolved_path and os.path.exists(resolved_path) and ONNX_AVAILABLE:
            try:
                avail = ort.get_available_providers()
                providers = [p for p in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider'] if p in avail] or ['CPUExecutionProvider']
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.enable_mem_pattern = True
                so.enable_cpu_mem_arena = True
                cpu_cnt = os.cpu_count() or 4
                so.intra_op_num_threads = max(1, cpu_cnt - 1) if cpu_cnt <= 4 else min(cpu_cnt // 2, 6)
                so.inter_op_num_threads = 1
                self.session = ort.InferenceSession(resolved_path, sess_options=so, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                inp_shape = self.session.get_inputs()[0].shape
                if len(inp_shape) == 4 and inp_shape[2] is not None and inp_shape[3] is not None:
                    self.input_shape = (int(inp_shape[2]), int(inp_shape[3]))
                print(f"✅ [YOLO ARROWS] Đã nạp thành công model: {resolved_path} (Kích thước: {self.input_shape})")
            except Exception as e:
                print(f"⚠️ [YOLO ARROWS] Không thể khởi tạo ONNX Session: {e}")
        else:
            print("ℹ️ [YOLO ARROWS] Chưa có model yolo_arrow.onnx. Sẽ sử dụng bộ nhận diện màu dự phòng (HSV).")

    def _resolve_model_path(self, model_path):
        candidates = []
        if model_path:
            candidates.append(model_path)
            if not os.path.isabs(model_path):
                candidates.append(os.path.join(os.path.expanduser('~'), 'robot_ws', model_path))
        
        # Các vị trí mặc định trong workspace (ưu tiên bao1.onnx vừa train)
        ws_models = os.path.join(os.path.expanduser('~'), 'robot_ws', 'models')
        candidates.append(os.path.join(ws_models, 'bao1.onnx'))
        candidates.append(os.path.join(ws_models, 'yolo_arrow.onnx'))
        candidates.append(os.path.join(ws_models, 'arrow_yolov8n.onnx'))
        candidates.append(os.path.join(ws_models, 'best_arrow.onnx'))

        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('my_robot_controller')
            candidates.append(os.path.join(share_dir, 'models', 'bao1.onnx'))
            candidates.append(os.path.join(share_dir, 'models', 'yolo_arrow.onnx'))
        except Exception:
            pass

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(current_dir, '..', 'models', 'bao1.onnx')))
        candidates.append(os.path.abspath(os.path.join(current_dir, '..', 'models', 'yolo_arrow.onnx')))

        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def reset_confirmation(self):
        """Xóa bộ đếm xác nhận theo thời gian (gọi khi chuyển trạng thái FSM nếu cần)."""
        self._hit_streak = {c: 0 for c in self.classes}
        self._last_detect_time = None

    def detect(self, bgr_image: np.ndarray, depth_image: np.ndarray = None) -> dict:
        """
        Nhận diện Mũi tên Vàng và Mũi tên Đen trên ảnh BGR.
        Trả về kết quả chi tiết kèm cự ly 3D và độ lệch tâm.

        'detected'     : đã qua ngưỡng tin cậy riêng của lớp VÀ xác nhận liên tiếp N lần.
        'raw_detected' : kết quả thô của khung hình hiện tại (chỉ qua ngưỡng tin cậy).
        """
        results = self._detect_single(bgr_image, depth_image)

        now = time.monotonic()
        if self._last_detect_time is None or (now - self._last_detect_time) > self.confirm_reset_sec:
            self._hit_streak = {c: 0 for c in self.classes}
        self._last_detect_time = now

        for cname in self.classes:
            r = results[cname]
            r['raw_detected'] = bool(r['detected'])
            self._hit_streak[cname] = self._hit_streak[cname] + 1 if r['detected'] else 0
            r['confirm_count'] = self._hit_streak[cname]
            r['detected'] = r['detected'] and self._hit_streak[cname] >= self.confirm_frames
        return results

    def _detect_single(self, bgr_image: np.ndarray, depth_image: np.ndarray = None) -> dict:
        """Nhận diện trên một khung hình duy nhất (chưa xác nhận theo thời gian)."""
        results = {
            'yellow_arrow': {'detected': False, 'confidence': 0.0, 'bbox': None, 'depth_m': float('inf'), 'offset_x': 0.0},
            'black_arrow':  {'detected': False, 'confidence': 0.0, 'bbox': None, 'depth_m': float('inf'), 'offset_x': 0.0},
        }

        if bgr_image is None:
            return results

        h_orig, w_orig = bgr_image.shape[:2]

        if self.session is not None:
            # ── 1. Suy luận qua model YOLO ONNX ───────────────────────────
            detections = self._infer_yolo(bgr_image)
            for det in detections:
                cid = det['class_id']
                if 0 <= cid < len(self.classes):
                    cname = self.classes[cid]
                    if det['confidence'] < self.class_conf_thresholds.get(cname, self.conf_threshold):
                        continue
                    bbox = det['bbox']
                    bw_box = max(1, bbox[2] - bbox[0])
                    bh_box = max(1, bbox[3] - bbox[1])
                    # Loại bỏ mẩu đuôi biển báo bị cắt cụt ở sát mép ảnh khi xe đã quay qua khỏi biển (bw/bh < 1.18)
                    if (bbox[0] <= 5 or bbox[2] >= w_orig - 5) and (float(bw_box) / float(bh_box) < 1.18):
                        continue
                    if det['confidence'] > results[cname]['confidence']:
                        cx = int((bbox[0] + bbox[2]) / 2.0)
                        cy = int((bbox[1] + bbox[3]) / 2.0)
                        
                        # Đo cự ly 3D từ Depth Map D435 trên vùng thân biển mũi tên (kèm đối chiếu quang học Pinhole)
                        depth_m = self._get_depth_at_point(
                            depth_image, cx, cy, bbox=bbox, img_w=w_orig, img_h=h_orig
                        )
                        
                        # Độ lệch tâm theo tỉ lệ (-1.0: lệch sát mép trái, +1.0: lệch sát mép phải, 0.0: ngay tim xe)
                        offset_x = (cx - (w_orig / 2.0)) / (w_orig / 2.0)

                        results[cname] = {
                            'detected': True,
                            'confidence': float(det['confidence']),
                            'bbox': bbox,
                            'center': (cx, cy),
                            'depth_m': depth_m,
                            'offset_x': float(offset_x)
                        }
        else:
            # ── 2. Bộ lọc màu dự phòng (HSV Color Fallback) khi chưa train YOLO ──
            results = self._detect_hsv_fallback(bgr_image, depth_image)

        return results

    def _infer_yolo(self, img: np.ndarray) -> list:
        h_orig, w_orig = img.shape[:2]
        target_h, target_w = self.input_shape

        # Resize trực tiếp về (640, 640) khớp 100% với bước tiền xử lý Stretch 640x640 của Roboflow
        resized = cv2.resize(img, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        outputs = self.session.run(None, {self.input_name: input_data})[0]
        # Đầu ra YOLOv8 / YOLO11 thường là (1, 4 + num_classes, num_anchors)
        if outputs.shape[1] < outputs.shape[2]:
            outputs = outputs[0].transpose(1, 0)
        else:
            outputs = outputs[0]

        boxes = []
        confidences = []
        class_ids = []

        x_scale = w_orig / float(target_w)
        y_scale = h_orig / float(target_h)

        for row in outputs:
            cx, cy, w, h = row[:4]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_threshold:
                x1 = int(np.clip((cx - w / 2.0) * x_scale, 0, w_orig - 1))
                y1 = int(np.clip((cy - h / 2.0) * y_scale, 0, h_orig - 1))
                bw = int(np.clip(w * x_scale, 1, w_orig - x1))
                bh = int(np.clip(h * y_scale, 1, h_orig - y1))

                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, bw, bh = boxes[idx]
                detections.append({
                    'class_id': class_ids[idx],
                    'confidence': confidences[idx],
                    'bbox': [x1, y1, x1 + bw, y1 + bh]
                })

        return detections

    def _get_depth_at_point(self, depth_image: np.ndarray, cx: int, cy: int,
                            bbox=None, img_w: int = 640, img_h: int = 480) -> float:
        """
        Đo cự ly quang học 3D (mét) từ Intel RealSense D435 (Aligned Depth 16UC1 mm hoặc 32FC1 m):
          1. Tự động quy đổi tỷ lệ tọa độ (cx, cy, bbox) nếu độ phân giải ảnh RGB và Depth khác nhau.
          2. Trích xuất vùng lõi trung tâm (Inner 50% Bounding Box) của biển mũi tên thay vì chỉ 1 điểm đơn lẻ,
             dùng phân vị tiền cảnh (30th percentile) để luôn lấy đúng khoảng cách mặt biển báo và KHÔNG bị
             lọt tia đo ra nền cỏ / tường cây ở xa phía sau biển báo.
          3. Đối chiếu chéo với khoảng cách quang học Pinhole (từ tiêu cự thực tế D435 fx/w = 615.66/640 = 0.962):
             nếu tia Depth bị vọt ra nền phía sau (> 1.35x cự ly quang học của khung biển), tự động kẹp về
             cự ly thực của biển báo để mốc dừng/rẽ 0.50m chính xác tuyệt đối từng cm.
        """
        pinhole_dist = None
        if bbox is not None and img_w > 0:
            bw = max(1.0, float(bbox[2] - bbox[0]))
            bw_ratio = bw / float(img_w)
            # Đã hiệu chuẩn theo tiêu cự D435 (fx/w = 0.962) & kích thước biển thực tế: bw_ratio = 0.23 <-> 0.50m
            pinhole_dist = float(np.clip(0.115 / bw_ratio, 0.20, 5.0))

        if depth_image is not None and depth_image.size > 0:
            dh, dw = depth_image.shape[:2]
            sx = float(dw) / float(max(1, img_w))
            sy = float(dh) / float(max(1, img_h))

            if bbox is not None:
                x1_d = int(np.clip(bbox[0] * sx, 0, dw - 1))
                y1_d = int(np.clip(bbox[1] * sy, 0, dh - 1))
                x2_d = int(np.clip(bbox[2] * sx, 0, dw - 1))
                y2_d = int(np.clip(bbox[3] * sy, 0, dh - 1))
                bw_d = max(1, x2_d - x1_d)
                bh_d = max(1, y2_d - y1_d)
                # Lấy vùng lõi 50% bên trong Bounding Box để bám chắc vào mặt phẳng biển mũi tên
                pad_x = int(bw_d * 0.25)
                pad_y = int(bh_d * 0.25)
                rx1 = max(0, min(dw - 1, x1_d + pad_x))
                rx2 = max(rx1 + 1, min(dw, x2_d - pad_x))
                ry1 = max(0, min(dh - 1, y1_d + pad_y))
                ry2 = max(ry1 + 1, min(dh, y2_d - pad_y))
                window = depth_image[ry1:ry2, rx1:rx2]
            else:
                cx_d = int(np.clip(cx * sx, 0, dw - 1))
                cy_d = int(np.clip(cy * sy, 0, dh - 1))
                y1, y2 = max(0, cy_d - 4), min(dh, cy_d + 5)
                x1, x2 = max(0, cx_d - 4), min(dw, cx_d + 5)
                window = depth_image[y1:y2, x1:x2]

            valid_m = None
            if window.dtype == np.uint16:
                valid = window[(window > 150) & (window < 4500)]
                if len(valid) > 0:
                    valid_m = valid.astype(np.float32) / 1000.0
            elif window.dtype in (np.float32, np.float64):
                valid = window[(window > 0.15) & (window < 4.5) & ~np.isnan(window)]
                if len(valid) > 0:
                    valid_m = valid.astype(np.float32)

            if valid_m is not None and len(valid_m) > 0:
                # Phân vị 30% giúp lấy đúng mặt phẳng biển mũi tên (tiền cảnh), loại bỏ tia lọt ra cỏ phía sau
                d_hw = float(np.percentile(valid_m, 30))
                if pinhole_dist is not None:
                    # Nếu tia hồng ngoại D435 bị xuyên qua mép biển ra nền cỏ xa phía sau (> 1.35x pinhole_dist),
                    # ưu tiên kết hợp với pinhole_dist để không bao giờ bị lố mốc 0.50m
                    if d_hw > pinhole_dist * 1.35:
                        return float(0.35 * d_hw + 0.65 * pinhole_dist)
                return d_hw

        if pinhole_dist is not None:
            return pinhole_dist

        return float('inf')

    def _detect_hsv_fallback(self, bgr_image: np.ndarray, depth_image: np.ndarray) -> dict:
        """Dự phòng nhận diện Mũi tên Vàng và Mũi tên Đen bằng OpenCV HSV + Hình thái học."""
        h_orig, w_orig = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        res = {
            'yellow_arrow': {'detected': False, 'confidence': 0.0, 'bbox': None, 'depth_m': float('inf'), 'offset_x': 0.0},
            'black_arrow':  {'detected': False, 'confidence': 0.0, 'bbox': None, 'depth_m': float('inf'), 'offset_x': 0.0},
        }

        # 1. Tìm Mũi tên Vàng: H: 16-35, S: 110-255, V: 110-255
        mask_yellow = cv2.inRange(hsv, np.array([16, 110, 110]), np.array([35, 255, 255]))
        mask_yellow[:int(0.35 * h_orig), :] = 0
        mask_yellow[int(0.92 * h_orig):, :] = 0
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        contours_y, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_y, best_y_area = None, 0
        for c in contours_y:
            area = cv2.contourArea(c)
            if area > 500:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(max(1, bh))
                if 1.5 <= aspect <= 6.5 and bw > 40:
                    if area > best_y_area:
                        best_y_area = area
                        best_y = (x, y, bw, bh)
        if best_y:
            x, y, bw, bh = best_y
            cx, cy = x + bw // 2, y + bh // 2
            bbox = [x, y, x + bw, y + bh]
            res['yellow_arrow'] = {
                'detected': True,
                'confidence': 0.85,
                'bbox': bbox,
                'center': (cx, cy),
                'depth_m': self._get_depth_at_point(depth_image, cx, cy, bbox=bbox, img_w=w_orig),
                'offset_x': float((cx - (w_orig / 2.0)) / (w_orig / 2.0))
            }

        # 2. Tìm Mũi tên Đen (Tách cọc sắt dọc bằng phép mở ngang)
        for thresh in [22, 28, 35]:
            mask_b = (gray <= thresh).astype(np.uint8) * 255
            mask_b[:int(0.45 * h_orig), :] = 0
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w_orig * 0.02)), 7))
            board = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k_open)
            board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)))
            contours_b, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_b, best_score = None, 0
            for c in contours_b:
                area = cv2.contourArea(c)
                if area > 800:
                    x, y, bw, bh = cv2.boundingRect(c)
                    aspect = bw / float(max(1, bh))
                    extent = area / float(max(1, bw * bh))
                    roi_mean = float(np.mean(gray[y:y+bh, x:x+bw]))
                    if 1.6 <= aspect <= 5.2 and extent > 0.60 and roi_mean < 38.0 and y > int(0.46 * h_orig):
                        score = area * extent * (42.0 - roi_mean)
                        if score > best_score:
                            best_score = score
                            best_b = (x, y, bw, bh)
            if best_b:
                x, y, bw, bh = best_b
                cx, cy = x + bw // 2, y + bh // 2
                bbox = [x, y, x + bw, y + bh]
                res['black_arrow'] = {
                    'detected': True,
                    'confidence': 0.82,
                    'bbox': bbox,
                    'center': (cx, cy),
                    'depth_m': self._get_depth_at_point(depth_image, cx, cy, bbox=bbox, img_w=w_orig),
                    'offset_x': float((cx - (w_orig / 2.0)) / (w_orig / 2.0))
                }
                break

        return res

