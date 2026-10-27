import os
import cv2
import numpy as np


class InferenceHandler:
    def __init__(
        self,
        model_path: str,
        mask_threshold: float = 0.35,
        input_size: tuple[int, int] = (384, 384),
        use_hsv_mask: bool = False,
        num_threads: int = 0
    ):
        self.model_path = model_path
        self.mask_threshold = mask_threshold  # kept for compatibility, used in binary segmentation
        self.input_size = input_size
        self.use_hsv_mask = use_hsv_mask
        self.num_threads = num_threads
        self.session = None
        self.input_name = None
        self.output_names = None
        self.latest_mask = None

        # Pre-allocated memory buffers for optimal throughput
        self._input_buffer = np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self._scale_inv_255 = np.float32(1.0 / 255.0)
        self._morph_kernel = np.ones((3, 3), dtype=np.uint8)

        if not self.use_hsv_mask:
            self.load_model()

    def load_model(self):
        import onnxruntime

        available = onnxruntime.get_available_providers()
        providers = []
        for ep in ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'OpenVINOExecutionProvider']:
            if ep in available:
                providers.append(ep)
        providers.append('CPUExecutionProvider')

        # Session optimization options
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True

        # CPU Thread configuration
        if self.num_threads > 0:
            so.intra_op_num_threads = self.num_threads
        else:
            cpu_cnt = os.cpu_count() or 4
            # On 4-core systems (e.g. Raspberry Pi), use 3 threads leaving 1 for ROS & LiDAR.
            # On 6-12+ core systems, cap at 6 to avoid diminishing returns.
            if cpu_cnt <= 4:
                so.intra_op_num_threads = max(1, cpu_cnt - 1)
            else:
                so.intra_op_num_threads = min(cpu_cnt // 2, 6)
        so.inter_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(self.model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [self.session.get_outputs()[0].name]

    def preprocess_image(self, bgr_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Converts BGR image to normalized 384x384 RGB tensor [0.0, 1.0] matching crop_row_cnn_best.onnx."""
        resized_bgr = cv2.resize(bgr_image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        float_img = rgb.astype(np.float32) * self._scale_inv_255
        self._input_buffer[0] = np.transpose(float_img, (2, 0, 1))
        return self._input_buffer, rgb

    def predict_mask(self, input_tensor: np.ndarray, enable_tta: bool = False) -> np.ndarray:
        """Runs model inference and returns probability mask [0.0, 1.0] (supports embedded Sigmoid or raw logits)."""
        ort_outs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_out = ort_outs[0]
        out_2d = raw_out[0, 0] if raw_out.ndim == 4 else np.squeeze(raw_out)

        # crop_row_cnn_best.onnx đã nhúng sẵn Sigmoid ở đầu ra [0.0, 1.0].
        # Chỉ áp dụng Sigmoid nếu mô hình trả về logits ngoài đoạn [0.0, 1.0], tránh lỗi Double-Sigmoid!
        if out_2d.min() >= -1e-4 and out_2d.max() <= 1.0001:
            probs = np.clip(out_2d, 0.0, 1.0)
        else:
            probs = 1.0 / (1.0 + np.exp(-np.clip(out_2d, -20.0, 20.0)))

        if enable_tta:
            input_tensor_flip = np.flip(input_tensor, axis=3)
            ort_outs_flip = self.session.run(self.output_names, {self.input_name: input_tensor_flip})
            raw_flip = ort_outs_flip[0]
            out_flip = raw_flip[0, 0] if raw_flip.ndim == 4 else np.squeeze(raw_flip)
            if out_flip.min() >= -1e-4 and out_flip.max() <= 1.0001:
                probs_flip = np.clip(out_flip, 0.0, 1.0)
            else:
                probs_flip = 1.0 / (1.0 + np.exp(-np.clip(out_flip, -20.0, 20.0)))
            probs_flip = np.flip(probs_flip, axis=1)
            probs = 0.5 * (probs + probs_flip)

        return probs

    # ------------------------------------------------------------------
    # Optimized Lane Tracking & Confidence
    # ------------------------------------------------------------------
    def find_lane_center(self, mask: np.ndarray) -> float:
        """Finds the X coordinate of the lane center using optimized row scanning."""
        if mask.ndim == 3:
            mask = mask[..., 0]

        h, w = mask.shape
        image_center = (w - 1) * 0.5

        # Bottom ROI (70% height to capture full carton boxes / crop rows)
        roi_ratio = 0.70
        y0 = int(h * (1.0 - roi_ratio))
        roi = mask[y0:, :]

        binary = (roi >= self.mask_threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)

        centers: list[float] = []
        min_lane_width = max(12, int(0.06 * w))
        center_idx = int(round(image_center))

        # Sample every 2nd row for 2x speedup with high fidelity at 512x512
        sampled_rows = binary[::2] if h >= 384 else binary

        for row in sampled_rows:
            cols = np.flatnonzero(row > 0)
            if cols.size < 2:
                continue

            # Prioritize left/right boundaries around center
            left_cols = cols[cols < center_idx]
            right_cols = cols[cols > center_idx]

            if left_cols.size > 0 and right_cols.size > 0:
                left = float(left_cols[-1])
                right = float(right_cols[0])
                if (right - left) >= min_lane_width:
                    centers.append((left + right) * 0.5)
                    continue

            # Fast run-length segment detection bounded by crops
            diffs = np.diff(np.pad(row.astype(np.int16), (1, 1), 'constant', constant_values=1))
            starts = np.where(diffs == -1)[0]
            ends = np.where(diffs == 1)[0] - 1
            widths = ends - starts + 1
            # Bắt buộc khoảng trống nằm giữa 2 cụm hàng và có bề rộng >= min_lane_width (loại bỏ khe hở nhiễu)
            valid_w = (starts > 0) & (ends < w - 1) & (widths >= min_lane_width)
            if np.any(valid_w):
                v_starts = starts[valid_w]
                v_ends = ends[valid_w]
                centers_x = (v_starts + v_ends) * 0.5
                dists = np.abs(centers_x - image_center)
                best_idx = np.argmin(dists)
                centers.append(float(centers_x[best_idx]))

        # Cần ít nhất 5 hàng quét tìm thấy lối đi 2 bên mới coi là tìm thấy luống hoàn chỉnh
        if len(centers) >= 5:
            return float(np.clip(np.median(centers), 0.0, w - 1.0))

        # Phân tích liên thông 2D (Connected Components Analysis) khi chỉ bắt được 1 hàng hoặc hàng chéo:
        # Bán bề rộng xe thực tế: ~0.30m (track width 0.58m, bánh xe 0.63m).
        # Hành lang 1.0m cần chừa tâm xe cách hàng cây ít nhất 0.50m (tương ứng ~0.38*w trên ảnh).
        half_lane_px = 0.38 * w  # Nửa bề rộng hành lang an toàn (~146px trên ảnh 384px, ~195px trên ảnh 512px)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Lọc bỏ các cụm nhiễu nhỏ (tự động co giãn theo độ phân giải h x w)
        area_scale = (h * w) / (512.0 * 512.0)
        min_comp_area = max(20, int(200 * area_scale))
        valid_comps = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_comp_area:
                pts = np.argwhere(labels == i)
                y_max = pts[:, 0].max()
                y_min = pts[:, 0].min()
                # Tọa độ chân hàng ở cận cảnh đáy ảnh (ngay trước mũi xe)
                bot_pts = pts[pts[:, 0] >= y_max - 25, 1]
                bot_x = float(bot_pts.mean()) if len(bot_pts) > 0 else float(centroids[i, 0])
                # Tọa độ ngọn hàng ở xa
                top_pts = pts[pts[:, 0] <= y_min + 25, 1]
                top_x = float(top_pts.mean()) if len(top_pts) > 0 else float(centroids[i, 0])
                valid_comps.append({
                    "area": area,
                    "cx": float(centroids[i, 0]),
                    "cy": float(centroids[i, 1]),
                    "bot_x": bot_x,
                    "top_x": top_x,
                    "dx_up": top_x - bot_x,
                    "bbox": stats[i]
                })

        if valid_comps:
            left_comps = [c for c in valid_comps if c["bot_x"] < center_idx]
            right_comps = [c for c in valid_comps if c["bot_x"] >= center_idx]

            # Trường hợp 1: Có các cụm rõ rệt nằm ở cả 2 bên tâm ảnh (Đủ 2 hàng Trái & Phải)
            if left_comps and right_comps:
                c_left = max(left_comps, key=lambda c: c["cx"])["cx"]
                c_right = min(right_comps, key=lambda c: c["cx"])["cx"]
                if (c_right - c_left) >= min_lane_width:
                    return float((c_left + c_right) * 0.5)

            # Trường hợp 2: Chỉ có 1 hàng đơn lẻ (hoặc hàng chéo chiếm ưu thế khi xe chạy sát mép)
            # Áp dụng phối hợp tọa độ chân hàng (bot_x), tâm (cx) và độ dốc phối cảnh (dx_up = top_x - bot_x):
            # - Hàng BÊN TRÁI: Thường nằm ở nửa trái, hoặc có độ dốc nghiêng sang phải về điểm tụ xa (dx_up > 0)
            #   => Lối đi giữa hàng nằm ở bên PHẢI hàng này (+ half_lane)
            # - Hàng BÊN PHẢI: Thường nằm ở nửa phải, hoặc có độ dốc nghiêng sang trái về điểm tụ xa (dx_up < 0)
            #   => Lối đi giữa hàng nằm ở bên TRÁI hàng này (- half_lane)
            main_comp = max(valid_comps, key=lambda c: c["area"])
            bot_x = main_comp["bot_x"]
            dx_up = main_comp["dx_up"]
            
            is_left_row = False
            if bot_x < 0.40 * w:
                is_left_row = True
            elif bot_x > 0.60 * w:
                is_left_row = False
            else:
                # Gốc hàng nằm ở vùng giữa (do xe đang đi chéo áp sát hàng):
                # Dùng hướng nghiêng phối cảnh về điểm tụ
                if dx_up > 3.0:
                    is_left_row = True   # Nghiêng [/] -> Hàng bên trái
                elif dx_up < -3.0:
                    is_left_row = False  # Nghiêng [\] -> Hàng bên phải
                else:
                    is_left_row = (main_comp["cx"] < center_idx)

            if is_left_row:
                target_center = bot_x + half_lane_px
            else:
                target_center = bot_x - half_lane_px

            return float(np.clip(target_center, 0.0, w - 1.0))

        return image_center

    def compute_vehicle_clearance_bias(self, mask: np.ndarray, max_angle_deg: float) -> float:
        """
        Tính toán góc bù trừ hao kích thước xe thật (Vehicle Footprint Clearance Bias)
        ở vùng cận cảnh (30% chiều cao đáy ảnh, ngay trước bánh xe và cản trước).
        
        Kích thước xe thật:
        - Bề rộng cầu bánh (track width): 0.58m, mép ngoài bánh xe: 0.63m.
        - Bán bề rộng xe từ tâm camera: ~0.315m.
        - Đệm an toàn sườn xe (margin): ~0.10m.
        - Tổng bán kính an toàn cần thiết: ~0.40m (~0.38*w trên FOV camera).
        - Ngưỡng nguy hiểm tối thiểu (mép bánh xe): ~0.28*w.
        """
        if mask.ndim == 3:
            mask = mask[..., 0]

        h, w = mask.shape
        image_center = (w - 1) * 0.5
        center_idx = int(round(image_center))

        # Quét vùng đáy ảnh (30% chiều cao cuối)
        y_start = int(h * 0.70)
        near_roi = mask[y_start:, :]
        binary = (near_roi >= self.mask_threshold).astype(np.uint8)

        safe_half_px = 0.38 * w   # Khoảng cách sườn an toàn (~146px trên 384x384)
        crit_half_px = 0.28 * w   # Ngưỡng mép bánh xe chạm cây (~107px trên 384x384)

        left_edges = []
        right_edges = []

        for row in binary[::2]:
            cols = np.flatnonzero(row > 0)
            if cols.size == 0:
                continue
            left_cols = cols[cols < center_idx]
            right_cols = cols[cols > center_idx]

            if left_cols.size > 0:
                left_edges.append(float(left_cols[-1]))
            if right_cols.size > 0:
                right_edges.append(float(right_cols[0]))

        bias_deg = 0.0
        med_left = np.median(left_edges) if len(left_edges) >= 3 else None
        med_right = np.median(right_edges) if len(right_edges) >= 3 else None

        # Trường hợp 1: Thấy cả 2 bên mép hàng ở cận cảnh
        if med_left is not None and med_right is not None:
            clearance_l = image_center - med_left
            clearance_r = med_right - image_center

            # Kiểm tra xem có bên nào vi phạm vùng an toàn thân xe không
            if clearance_l < safe_half_px or clearance_r < safe_half_px:
                diff = clearance_r - clearance_l
                total_span = max(1.0, clearance_l + clearance_r)
                ratio = diff / total_span
                bias_deg = ratio * max_angle_deg * 1.5

                # Nguy hiểm cận kề bánh xe: cưỡng bức bẻ lái thoát hiểm
                if clearance_l < crit_half_px:
                    crit_pen = (crit_half_px - clearance_l) / crit_half_px
                    bias_deg = max(bias_deg, crit_pen * max_angle_deg)
                elif clearance_r < crit_half_px:
                    crit_pen = (crit_half_px - clearance_r) / crit_half_px
                    bias_deg = min(bias_deg, -crit_pen * max_angle_deg)

        # Trường hợp 2: Chỉ phát hiện mép hàng bên TRÁI ở cận cảnh
        elif med_left is not None:
            clearance_l = image_center - med_left
            if clearance_l < safe_half_px:
                pen = (safe_half_px - clearance_l) / safe_half_px
                bias_deg = pen * max_angle_deg * 1.2
                if clearance_l < crit_half_px:
                    bias_deg = max(bias_deg, 0.75 * max_angle_deg)

        # Trường hợp 3: Chỉ phát hiện mép hàng bên PHẢI ở cận cảnh
        elif med_right is not None:
            clearance_r = med_right - image_center
            if clearance_r < safe_half_px:
                pen = (safe_half_px - clearance_r) / safe_half_px
                bias_deg = -pen * max_angle_deg * 1.2
                if clearance_r < crit_half_px:
                    bias_deg = min(bias_deg, -0.75 * max_angle_deg)

        return float(np.clip(bias_deg, -max_angle_deg, max_angle_deg))

    def compute_steering_angle(self, mask: np.ndarray, max_angle_deg: float = 3.5) -> float:
        """Translates offset of lane center to steering angle (in degrees), factoring in vehicle footprint."""
        _, w = mask.shape[:2]
        lane_center = self.find_lane_center(mask)
        image_center = (w - 1) * 0.5

        offset = (lane_center - image_center) / max(image_center, 1.0)
        base_angle = float(np.clip(offset * max_angle_deg, -max_angle_deg, max_angle_deg))
        clearance_bias = self.compute_vehicle_clearance_bias(mask, max_angle_deg)
        return float(np.clip(base_angle + clearance_bias, -max_angle_deg, max_angle_deg))

    def compute_row_confidence(self, mask_prob: np.ndarray, roi_ratio: float = 0.70) -> float:
        """Computes a confidence score based on crop row lane detection density."""
        if mask_prob.ndim == 3:
            mask_prob = mask_prob[..., 0]
            
        h, w = mask_prob.shape
        y0 = int(h * (1.0 - roi_ratio))
        roi = mask_prob[y0:, :]
        
        binary = (roi >= self.mask_threshold).astype(np.uint8)
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)
        
        valid_rows = 0
        sampled_rows = binary_closed[::2] if h >= 384 else binary_closed
        total_rows = sampled_rows.shape[0]
        center_idx = int(w * 0.5)
        min_lane_width = max(12, int(0.06 * w))
        
        for row in sampled_rows:
            cols = np.flatnonzero(row > 0)
            if cols.size < 2:
                continue
                
            left_cols = cols[cols < center_idx]
            right_cols = cols[cols > center_idx]
            
            # Check 1: Lane contains the image center
            if left_cols.size > 0 and right_cols.size > 0:
                if (float(right_cols[0]) - float(left_cols[-1])) >= min_lane_width:
                    valid_rows += 1
                    continue
            
            # Check 2: Shifted lane
            diffs = np.diff(np.pad(row.astype(np.int16), (1, 1), 'constant', constant_values=1))
            starts = np.where(diffs == -1)[0]
            ends = np.where(diffs == 1)[0] - 1
            valid = (starts > 0) & (ends < w - 1)
            if np.any(valid):
                widths = ends[valid] - starts[valid] + 1
                if np.max(widths) >= min_lane_width:
                    valid_rows += 1
                    
        # Confidence Score: 1.0 if valid lane is detected on >50% of ROI rows
        row_score = valid_rows / (total_rows * 0.5 + 1e-6)
        confidence = float(np.clip(row_score, 0.0, 1.0))

        # Hỗ trợ bám 1 hàng (Single-Row Confidence):
        # Nếu chỉ thấy 1 vạch hàng cây/thùng (chưa đủ 2 hàng hai bên), cấp confidence trong dải 0.32 ~ 0.38
        # để xe vẫn duy trì bám luống (>= low_confidence_threshold 0.30) nhưng KHÔNG vượt ngưỡng 0.45
        # của pha khóa 2 hàng miệng Luống 2 (TURN_90_INTO_ROW2).
        if confidence < 0.35:
            area_scale = (h * w) / (512.0 * 512.0)
            min_single_area = max(80, int(450 * area_scale))
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_closed)
            max_area = max([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)], default=0)
            if max_area >= min_single_area:
                single_score = min(0.38, 0.32 + (max_area / max(1.0, 3000.0 * area_scale)) * 0.06)
                confidence = max(confidence, single_score)

        return confidence

    def process_image(self, bgr_image: np.ndarray, max_angle_deg: float = 3.5) -> tuple[float, float, float, float]:
        """
        Runs full pre-processing, CNN inference (or HSV segmentation), lane center calculation,
        and confidence estimation.
        Returns (heading_error, lane_offset, lane_center, confidence).
        """
        if self.use_hsv_mask:
            # Color-based segmentation for simulation
            hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 30, 30])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask_prob = (mask * self._scale_inv_255).astype(np.float32)
            mask_prob = cv2.resize(mask_prob, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
            confidence = self.compute_row_confidence(mask_prob)
        else:
            input_tensor, _ = self.preprocess_image(bgr_image)
            mask_prob = self.predict_mask(input_tensor, enable_tta=False)
            confidence = self.compute_row_confidence(mask_prob)
            
            # Domain Adaptation Check: Only fallback to HSV green segmentation in simulation (when use_hsv_mask is enabled)
            # In real world, ground is green grass and boxes are white/cardboard, so never overwrite CNN with green HSV
            if self.use_hsv_mask and confidence < 0.30:
                hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 30, 30])
                upper_green = np.array([85, 255, 255])
                mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
                hsv_prob = (mask_hsv * self._scale_inv_255).astype(np.float32)
                hsv_prob = cv2.resize(hsv_prob, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
                
                hsv_conf = self.compute_row_confidence(hsv_prob)
                if hsv_conf > confidence:
                    mask_prob = hsv_prob
                    confidence = hsv_conf

        # Lane center and heading error factoring in vehicle footprint clearance
        lane_center = self.find_lane_center(mask_prob)
        _, w = mask_prob.shape[:2]
        image_center = (w - 1) * 0.5
        lane_offset = (lane_center - image_center) / max(image_center, 1.0)
        base_heading_error = float(np.clip(lane_offset * max_angle_deg, -max_angle_deg, max_angle_deg))
        
        # Trừ hao kích thước xe thật ở cận cảnh (Vehicle Footprint Clearance Bias)
        clearance_bias = self.compute_vehicle_clearance_bias(mask_prob, max_angle_deg)
        heading_error = float(np.clip(base_heading_error + clearance_bias, -max_angle_deg, max_angle_deg))
        
        self.latest_mask = mask_prob
        return heading_error, lane_offset, lane_center, confidence
