#!/usr/bin/env python3
"""
Telemetry and Event Logger Utility for ROS 2 LuanVan Robot.
Exports telemetry metrics (FSM, Pose, Steering Angle, Velocities, GPS) to structured CSV and formatted text log files.
"""

import os
import time
import csv
from datetime import datetime


class TelemetryLogger:
    def __init__(self, log_dir=None, enable_csv=True, enable_text_log=True):
        if log_dir is None or not os.path.isabs(log_dir):
            # Resolve workspace root
            ws_dir = os.environ.get('WS_DIR', '')
            if not ws_dir or not os.path.isdir(ws_dir):
                # Search upward from __file__
                curr = os.path.abspath(__file__)
                for _ in range(6):
                    parent = os.path.dirname(curr)
                    if os.path.isdir(os.path.join(parent, 'src')) or os.path.isdir(os.path.join(parent, 'install')):
                        ws_dir = parent
                        break
                    curr = parent
            if not ws_dir or not os.path.isdir(ws_dir):
                for candidate in [
                    '/home/vinh/Màn hình nền/robot_ws',
                    os.path.expanduser('~/robot_ws'),
                    os.path.expanduser('~/robot-ws')
                ]:
                    if os.path.isdir(candidate):
                        ws_dir = candidate
                        break
            if not ws_dir or not os.path.isdir(ws_dir):
                ws_dir = os.path.expanduser('~')

            sub_dir = log_dir if (log_dir and log_dir != 'logs') else 'logs'
            self.log_dir = os.path.join(ws_dir, sub_dir)
        else:
            self.log_dir = os.path.abspath(os.path.expanduser(log_dir))

        self.enable_csv = enable_csv
        self.enable_text_log = enable_text_log
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_filename = os.path.join(self.log_dir, f"telemetry_{timestamp_str}.csv")
        self.log_filename = os.path.join(self.log_dir, f"system_run_{timestamp_str}.log")
        self.latest_csv_filename = os.path.join(self.log_dir, "latest_telemetry.csv")
        self.latest_log_filename = os.path.join(self.log_dir, "latest_system_run.log")
        
        self.csv_headers = [
            'Time',
            'Unix_Timestamp',
            'FSM_State',
            'Pos_X_m',
            'Pos_Y_m',
            'Yaw_rad',
            'Steer_Angle_deg',
            'Raw_Steer_deg',
            'Lane_Offset_m',
            'Linear_Vel_mps',
            'Angular_Vel_radps',
            'IMU_Yaw_rad',
            'IMU_Angular_Vel_z',
            'IMU_Accel_x',
            'Confidence',
            'Inference_ms',
            'FPS',
            'Dist_Traveled_m',
            'GPS_Latitude',
            'GPS_Longitude',
            'GPS_Altitude_m',
            'GPS_DMS',
            'GPS_Status',
            'Event'
        ]

        self.turn_scores_csv = os.path.join(self.log_dir, "omega_turn_scores.csv")
        self.turn_score_headers = [
            'Time', 'Unix_Timestamp', 'Total_Score', 'Grade', 'Grade_Desc',
            'Duration_s', 'Rel_dY_m', 'Err_dY_m', 'Score_dY',
            'Rel_Yaw_deg', 'Err_Yaw_deg', 'Score_Yaw',
            'Avg_Slip_pct', 'Score_Slip',
            'CNN_Conf', 'Score_Vision',
            'RMSE_CTE_m', 'Num_Samples', 'Num_GPS_Samples'
        ]

        if self.enable_csv and not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    f.flush()
                self._update_symlink(self.csv_filename, self.latest_csv_filename)
            except Exception as e:
                print(f"[TelemetryLogger Error] Failed to initialize CSV: {e}")

        if self.enable_csv and not os.path.exists(self.turn_scores_csv):
            try:
                with open(self.turn_scores_csv, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.turn_score_headers)
                    f.flush()
            except Exception as e:
                print(f"[TelemetryLogger Error] Failed to initialize turn scores CSV: {e}")

        if self.enable_text_log:
            start_header = (
                "=" * 80 + "\n"
                f" ROBOT SYSTEM RUN LOG | STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f" CSV Telemetry File : {self.csv_filename}\n"
                f" Text System Log File: {self.log_filename}\n"
                f" Latest CSV Link    : {self.latest_csv_filename}\n"
                f" Latest Text Link   : {self.latest_log_filename}\n"
                + "=" * 80 + "\n"
            )
            try:
                with open(self.log_filename, mode='w') as f:
                    f.write(start_header)
                    f.flush()
                self._update_symlink(self.log_filename, self.latest_log_filename)
            except Exception as e:
                print(f"[TelemetryLogger Error] Failed to write header: {e}")

            self.log_event("SYSTEM_START", "Telemetry and Logging system started cleanly.")

    def _update_symlink(self, src: str, dst: str):
        """Creates or updates a symlink or copy pointing to the latest log file."""
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.basename(src), dst)
        except Exception:
            try:
                import shutil
                shutil.copyfile(src, dst)
            except Exception:
                pass

    def log_telemetry(self, data: dict):
        """
        Appends a clean, formatted telemetry record to the CSV log file.
        """
        if not self.enable_csv:
            return

        now_sec = time.time()
        time_str = datetime.fromtimestamp(now_sec).strftime("%H:%M:%S.%f")[:-3]

        row = [
            time_str,
            f"{now_sec:.3f}",
            data.get('fsm_state', 'UNKNOWN'),
            f"{data.get('x', 0.0):.3f}",
            f"{data.get('y', 0.0):.3f}",
            f"{data.get('yaw', 0.0):.3f}",
            f"{data.get('steering_angle_deg', 0.0):.2f}",
            f"{data.get('raw_steer_deg', 0.0):.2f}",
            f"{data.get('lane_offset', 0.0):.3f}",
            f"{data.get('linear_velocity', 0.0):.3f}",
            f"{data.get('angular_velocity', 0.0):.3f}",
            f"{data.get('imu_yaw', 0.0):.3f}",
            f"{data.get('imu_angular_vel_z', 0.0):.3f}",
            f"{data.get('imu_accel_x', 0.0):.3f}",
            f"{data.get('confidence', 0.0):.3f}",
            f"{data.get('inference_ms', 0.0):.1f}",
            f"{data.get('fps', 0.0):.1f}",
            f"{data.get('distance_traveled', 0.0):.2f}",
            f"{data.get('gps_latitude', 0.0):.8f}",
            f"{data.get('gps_longitude', 0.0):.8f}",
            f"{data.get('gps_altitude', 0.0):.2f}",
            data.get('gps_dms', ''),
            data.get('gps_status', 'NO_FIX'),
            data.get('event', '')
        ]

        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"[TelemetryLogger Error] Failed to write CSV: {e}")

    def log_event(self, event_type: str, message: str):
        """
        Logs a key system event to the text log file with clear time separation.
        """
        if not self.enable_text_log:
            return

        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{time_str}] [{event_type:^18s}] {message}\n"
        
        try:
            with open(self.log_filename, mode='a') as f:
                f.write(log_line)
                f.flush()
        except Exception as e:
            print(f"[TelemetryLogger Error] Failed to write text log: {e}")

    def log_turn_evaluation(self, eval_result: dict):
        """
        Ghi nhận kết quả đánh giá quay đầu Omega Turn vào tệp omega_turn_scores.csv.
        """
        if not self.enable_csv or not eval_result:
            return

        now_sec = time.time()
        time_str = datetime.fromtimestamp(now_sec).strftime("%H:%M:%S.%f")[:-3]
        sb = eval_result.get('score_breakdown', {})

        row = [
            time_str,
            f"{now_sec:.3f}",
            f"{eval_result.get('total_score', 0.0):.1f}",
            eval_result.get('grade', 'F'),
            eval_result.get('grade_desc', 'UNKNOWN'),
            f"{eval_result.get('duration_s', 0.0):.1f}",
            f"{eval_result.get('final_rel_y', 0.0):.3f}",
            f"{eval_result.get('err_dy_m', 0.0):.3f}",
            f"{sb.get('score_dy', 0.0):.1f}",
            f"{eval_result.get('final_rel_yaw_deg', 0.0):.1f}",
            f"{eval_result.get('err_yaw_deg', 0.0):.1f}",
            f"{sb.get('score_yaw', 0.0):.1f}",
            f"{eval_result.get('avg_slip_percent', 0.0):.1f}",
            f"{sb.get('score_slip', 0.0):.1f}",
            f"{eval_result.get('cnn_confidence', 0.0):.3f}",
            f"{sb.get('score_vision', 0.0):.1f}",
            f"{eval_result.get('rmse_cte_m', 0.0):.3f}",
            eval_result.get('num_samples', 0),
            eval_result.get('num_gps_samples', 0),
        ]

        try:
            with open(self.turn_scores_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"[TelemetryLogger Error] Failed to write turn scores CSV: {e}")


def main(args=None):
    logger = TelemetryLogger()
    print(f"TelemetryLogger initialized.")
    print(f"📁 Log Directory: {logger.log_dir}")
    print(f"📄 CSV Log      : {logger.csv_filename}")
    print(f"📄 Text Log     : {logger.log_filename}")
    print(f"🔗 Latest CSV   : {logger.latest_csv_filename}")
    print(f"🔗 Latest Text  : {logger.latest_log_filename}")
    logger.log_event("SYSTEM", "TelemetryLogger test event recorded successfully.")
    print("✅ Logged test event successfully.")


if __name__ == '__main__':
    main()
