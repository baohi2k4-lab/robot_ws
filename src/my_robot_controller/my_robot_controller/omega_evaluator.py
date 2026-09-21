#!/usr/bin/env python3
"""
Omega Turn Performance Evaluator for Agricultural Autonomous Robot.
Evaluates turn maneuvers using Relative Local Frame, GPS Doppler velocity,
and CNN Handover metrics tailored for u-blox NEO-M10 (without RTK).
"""

import math
import time
from typing import Dict, Any, Optional, List, Tuple


def wrap_to_pi(angle_rad: float) -> float:
    """Wraps an angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class OmegaTurnEvaluator:
    """
    Bộ đánh giá & chấm điểm kỹ thuật cho quỹ đạo quay đầu Omega Turn:
      1. Tọa độ tương đối cục bộ (Relative Local Frame): Chốt gốc (X0, Y0, Yaw0) lúc rẽ.
      2. Tỷ lệ trượt bánh vi sai (Wheel Slip Ratio): So sánh Doppler GPS (M10) vs Wheel Odom.
      3. Sai số góc xoay đảo hướng (Yaw Error vs 180°).
      4. Sai số dịch chuyển ngang (Lateral Shift vs row_spacing).
      5. Khả năng chuyển giao Vision (CNN Handover Confidence).
    """

    def __init__(self):
        self.is_evaluating: bool = False
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        # Mốc gốc tọa độ cục bộ ban đầu
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.yaw0: float = 0.0

        # Cấu hình mục tiêu
        self.target_dy: float = -1.20       # m (Mặc định -1.20m cho rẽ phải)
        self.target_yaw_deg: float = 180.0  # độ (Đảo chiều 180 độ)
        self.reference_waypoints: Optional[List[Tuple[float, float, float]]] = None

        # Danh sách mẫu thu thập
        self.rel_x_samples: List[float] = []
        self.rel_y_samples: List[float] = []
        self.rel_yaw_deg_samples: List[float] = []
        self.v_odom_samples: List[float] = []
        self.v_gps_samples: List[float] = []
        self.slip_samples: List[float] = []
        self.cte_samples: List[float] = []

        # Kết quả đánh giá gần nhất
        self.latest_result: Optional[Dict[str, Any]] = None

    def start_turn(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        target_dy: float = -1.20,
        target_yaw_deg: float = 180.0,
        reference_waypoints: Optional[List[Any]] = None
    ):
        """Kích hoạt phiên đánh giá quay đầu và chốt mốc tọa độ cục bộ ban đầu."""
        self.is_evaluating = True
        self.start_time = time.time()
        self.end_time = 0.0

        self.x0 = float(start_x)
        self.y0 = float(start_y)
        self.yaw0 = float(start_yaw)
        self.target_dy = float(target_dy)
        self.target_yaw_deg = float(target_yaw_deg)

        # Trích xuất waypoints tham chiếu dạng (x, y, yaw) nếu có
        self.reference_waypoints = []
        if reference_waypoints:
            for wp in reference_waypoints:
                if hasattr(wp, 'x') and hasattr(wp, 'y'):
                    self.reference_waypoints.append((float(wp.x), float(wp.y), getattr(wp, 'yaw', 0.0)))
                elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                    yaw = float(wp[2]) if len(wp) >= 3 else 0.0
                    self.reference_waypoints.append((float(wp[0]), float(wp[1]), yaw))

        self.rel_x_samples.clear()
        self.rel_y_samples.clear()
        self.rel_yaw_deg_samples.clear()
        self.v_odom_samples.clear()
        self.v_gps_samples.clear()
        self.slip_samples.clear()
        self.cte_samples.clear()
        self.latest_result = None

    def update_step(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        v_odom: float,
        v_gps: Optional[float] = None
    ):
        """
        Ghi nhận một mẫu trạng thái trong chu trình quay đầu.
        Chuyển đổi tức thời sang hệ quy chiếu cục bộ (Relative Local Frame).
        """
        if not self.is_evaluating:
            return

        # 1. Chuyển đổi sang hệ quy chiếu tương đối (Local Frame tại thời điểm t0)
        dx = current_x - self.x0
        dy = current_y - self.y0
        cos0 = math.cos(self.yaw0)
        sin0 = math.sin(self.yaw0)

        x_rel = dx * cos0 + dy * sin0
        y_rel = -dx * sin0 + dy * cos0
        yaw_rel = wrap_to_pi(current_yaw - self.yaw0)
        yaw_rel_deg = math.degrees(abs(yaw_rel))

        self.rel_x_samples.append(x_rel)
        self.rel_y_samples.append(y_rel)
        self.rel_yaw_deg_samples.append(yaw_rel_deg)
        self.v_odom_samples.append(abs(v_odom))

        # 2. Đo lường vận tốc Doppler và tỷ lệ trượt bánh vi sai
        if v_gps is not None and not math.isnan(v_gps) and v_gps >= 0.0:
            self.v_gps_samples.append(v_gps)
            # Chỉ tính tỷ lệ trượt khi xe đang lăn bánh có vận tốc (> 0.02 m/s)
            if abs(v_odom) > 0.02:
                slip = abs(1.0 - (v_gps / abs(v_odom)))
                # Giới hạn trượt trong khoảng hợp lý [0.0, 1.0]
                slip = min(1.0, max(0.0, slip))
                self.slip_samples.append(slip)

        # 3. Tính khoảng cách vuông góc ngắn nhất (CTE) tới quỹ đạo tham chiếu nếu có
        if self.reference_waypoints and len(self.reference_waypoints) > 0:
            min_d_sq = float('inf')
            for wx, wy, _ in self.reference_waypoints:
                d_sq = (current_x - wx) ** 2 + (current_y - wy) ** 2
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
            self.cte_samples.append(math.sqrt(min_d_sq))

    def finalize_evaluation(
        self,
        final_confidence: float = 0.0,
        final_x: Optional[float] = None,
        final_y: Optional[float] = None,
        final_yaw: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Kết thúc chu trình quay đầu, tính toán điểm số tổng hợp và xếp loại.
        """
        self.end_time = time.time()
        self.is_evaluating = False

        duration = max(0.1, self.end_time - self.start_time)

        # Tọa độ tương đối cuối cùng
        if final_x is not None and final_y is not None and final_yaw is not None:
            dx = final_x - self.x0
            dy = final_y - self.y0
            cos0 = math.cos(self.yaw0)
            sin0 = math.sin(self.yaw0)
            final_rel_x = dx * cos0 + dy * sin0
            final_rel_y = -dx * sin0 + dy * cos0
            final_rel_yaw_deg = math.degrees(abs(wrap_to_pi(final_yaw - self.yaw0)))
        elif len(self.rel_x_samples) > 0:
            final_rel_x = self.rel_x_samples[-1]
            final_rel_y = self.rel_y_samples[-1]
            final_rel_yaw_deg = self.rel_yaw_deg_samples[-1]
        else:
            final_rel_x = 0.0
            final_rel_y = 0.0
            final_rel_yaw_deg = 0.0

        # ── 1. Sai số dịch chuyển ngang (Delta Y vs target_dy) ───────────
        err_dy = abs(final_rel_y - self.target_dy)
        # Dung sai 0.25m: err <= 0.05m -> 100đ, err >= 0.25m -> 0đ
        score_dy = max(0.0, min(100.0, 100.0 * (1.0 - (err_dy / 0.25))))

        # ── 2. Sai số góc xoay đảo hướng (Delta Yaw vs 180°) ─────────────
        err_yaw = abs(final_rel_yaw_deg - self.target_yaw_deg)
        # Dung sai 15°: err <= 3° -> 100đ, err >= 15° -> 0đ
        score_yaw = max(0.0, min(100.0, 100.0 * (1.0 - (err_yaw / 15.0))))

        # ── 3. Tỷ lệ trượt bánh vi sai (Doppler Slip Ratio) ───────────────
        if len(self.slip_samples) > 0:
            avg_slip = float(sum(self.slip_samples) / len(self.slip_samples))
        else:
            # Nếu không có mẫu GPS Doppler, đánh giá qua tỷ lệ dịch chuyển odom
            avg_slip = 0.05  # Giả định mặc định trượt nhẹ 5%
        # Trượt <= 5% -> 100đ, trượt >= 30% -> 0đ
        score_slip = max(0.0, min(100.0, 100.0 * (1.0 - (avg_slip / 0.30))))

        # ── 4. Khả năng chuyển giao Vision (CNN Handover) ────────────────
        # Conf >= 0.65 -> 100đ, Conf <= 0.25 -> 0đ
        conf_clamped = max(0.0, min(1.0, float(final_confidence)))
        score_vision = max(0.0, min(100.0, ((conf_clamped - 0.25) / 0.40) * 100.0))

        # ── 5. Sai số bám quỹ đạo trung bình (RMSE CTE) ───────────────────
        rmse_cte = 0.0
        if len(self.cte_samples) > 0:
            rmse_cte = math.sqrt(sum(c ** 2 for c in self.cte_samples) / len(self.cte_samples))

        # ── TỔNG ĐIỂM CÓ TRỌNG SỐ (Thang điểm 100) ────────────────────────
        # Trọng số: 30% dY, 25% Yaw, 25% Slip, 20% Vision Handover
        total_score = (
            0.30 * score_dy +
            0.25 * score_yaw +
            0.25 * score_slip +
            0.20 * score_vision
        )
        total_score = round(max(0.0, min(100.0, total_score)), 1)

        # Xếp loại
        if total_score >= 88.0:
            grade = "A"
            grade_desc = "XUẤT SẮC"
        elif total_score >= 70.0:
            grade = "B"
            grade_desc = "ĐẠT"
        elif total_score >= 50.0:
            grade = "C"
            grade_desc = "CẦN TINH CHỈNH"
        else:
            grade = "F"
            grade_desc = "KHÔNG ĐẠT"

        result = {
            "total_score": total_score,
            "grade": grade,
            "grade_desc": grade_desc,
            "duration_s": round(duration, 1),
            "final_rel_x": round(final_rel_x, 3),
            "final_rel_y": round(final_rel_y, 3),
            "final_rel_yaw_deg": round(final_rel_yaw_deg, 1),
            "err_dy_m": round(err_dy, 3),
            "err_yaw_deg": round(err_yaw, 1),
            "avg_slip_percent": round(avg_slip * 100.0, 1),
            "cnn_confidence": round(conf_clamped, 3),
            "rmse_cte_m": round(rmse_cte, 3),
            "score_breakdown": {
                "score_dy": round(score_dy, 1),
                "score_yaw": round(score_yaw, 1),
                "score_slip": round(score_slip, 1),
                "score_vision": round(score_vision, 1),
            },
            "num_samples": len(self.rel_x_samples),
            "num_gps_samples": len(self.v_gps_samples),
        }

        self.latest_result = result
        return result

    def get_summary_text(self) -> str:
        """Tạo chuỗi báo cáo chuẩn hóa in ra terminal."""
        if not self.latest_result:
            return "Chưa có kết quả đánh giá quay đầu."

        r = self.latest_result
        sb = r["score_breakdown"]
        lines = [
            "=" * 70,
            f" 🎯 BÁO CÁO CHẤM ĐIỂM QUAY ĐẦU OMEGA TURN (NEO-M10 & VISION)",
            "=" * 70,
            f" ▶ TỔNG ĐIỂM: {r['total_score']:.1f}/100  |  XẾP LOẠI: HẠNG {r['grade']} ({r['grade_desc']})",
            f" ▶ Thời gian thực thi: {r['duration_s']:.1f}s | Số mẫu: {r['num_samples']} (Doppler GPS: {r['num_gps_samples']})",
            "-" * 70,
            f" 1. Dịch chuyển ngang (ΔY): {r['final_rel_y']:+.2f}m (Mục tiêu: {self.target_dy:+.2f}m | Lệch: {r['err_dy_m']*100:.1f}cm) -> {sb['score_dy']:.1f} đ (Trọng số 30%)",
            f" 2. Góc đảo hướng (ΔYaw) : {r['final_rel_yaw_deg']:.1f}° (Mục tiêu: {self.target_yaw_deg:.0f}° | Lệch: {r['err_yaw_deg']:.1f}°) -> {sb['score_yaw']:.1f} đ (Trọng số 25%)",
            f" 3. Chống trượt bánh (Slip): {r['avg_slip_percent']:.1f}% (Doppler vs Odom)            -> {sb['score_slip']:.1f} đ (Trọng số 25%)",
            f" 4. Chuyển giao Vision   : CNN Conf = {r['cnn_confidence']:.2f}                       -> {sb['score_vision']:.1f} đ (Trọng số 20%)",
            "=" * 70,
        ]
        return "\n".join(lines)

    def get_hud_string(self) -> str:
        """Tạo chuỗi hiển thị ngắn gọn cho HUD Marker trên RViz."""
        if not self.latest_result:
            return ""
        r = self.latest_result
        return (
            f"[DIEM OMEGA: {r['total_score']:.0f}/100 (HANG {r['grade']})] | "
            f"Slip: {r['avg_slip_percent']:.1f}% | "
            f"dY: {r['final_rel_y']:+.2f}m | "
            f"Yaw: {r['final_rel_yaw_deg']:.0f}° | "
            f"CNN: {r['cnn_confidence']:.2f}"
        )
