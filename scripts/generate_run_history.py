#!/usr/bin/env python3
"""
Sổ Nhật Ký & Theo Dõi Lịch Sử Các Lần Chạy Robot (Run History & Session Logger).
Tự động tổng hợp dữ liệu từ các file telemetry_*.csv:
- Ngày giờ & Thời lượng chạy
- Quãng đường di chuyển (m) & Vận tốc trung bình (m/s)
- Độ tin cậy trung bình của AI CNN (%)
- Trạng thái quay đầu Omega Turn (Thành công / Dở dang)
- Kết quả nhiệm vụ (Hoàn thành Goal 3 / Dừng tay Ctrl+C / Vật cản)
Xuất ra 2 định dạng:
1. logs/lich_su_chay_xe.csv (Dữ liệu bảng phân tích / Excel)
2. logs/lich_su_chay_xe.md (Bảng Markdown hiển thị đẹp mắt)
"""

import os
import sys
import csv
import glob
from datetime import datetime

WS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_DIR = os.path.join(WS_DIR, 'logs')

CSV_HISTORY_FILE = os.path.join(LOGS_DIR, 'lich_su_chay_xe.csv')
MD_HISTORY_FILE = os.path.join(LOGS_DIR, 'lich_su_chay_xe.md')

HEADERS = [
    'STT',
    'Thoi_Gian_Bat_Dau',
    'Thoi_Luong_s',
    'Quang_Duong_m',
    'Van_Toc_Max_mps',
    'Van_Toc_TB_mps',
    'CNN_Confidence_TB',
    'Quay_Dau_Omega',
    'Ket_Qua_Nhiem_Vu',
    'File_Telemetry'
]

def analyze_single_telemetry(csv_path):
    """Phân tích 1 tệp telemetry_*.csv để trích xuất các chỉ số tổng kết."""
    filename = os.path.basename(csv_path)
    # Lấy timestamp từ tên file
    # Format: telemetry_YYYY-MM-DD_HH-MM-SS.csv
    try:
        ts_part = filename.replace('telemetry_', '').replace('.csv', '')
        start_dt = datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S")
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        start_str = filename

    rows = []
    try:
        with open(csv_path, mode='r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return None

    if not rows or len(rows) < 2:
        return {
            'Thoi_Gian_Bat_Dau': start_str,
            'Thoi_Luong_s': 0.0,
            'Quang_Duong_m': 0.0,
            'Van_Toc_Max_mps': 0.0,
            'Van_Toc_TB_mps': 0.0,
            'CNN_Confidence_TB': 0.0,
            'Quay_Dau_Omega': 'CHƯA_CHẠY',
            'Ket_Qua_Nhiem_Vu': 'KHỞI_ĐỘNG_NGAY_LẬP_TỨC',
            'File_Telemetry': filename
        }

    # Thời lượng
    try:
        t0 = float(rows[0].get('Unix_Timestamp') or 0)
        t1 = float(rows[-1].get('Unix_Timestamp') or 0)
        duration = max(0.0, t1 - t0)
    except Exception:
        duration = len(rows) * 0.067

    # Vận tốc
    v_list = []
    for r in rows:
        try:
            v_val = abs(float(r.get('Linear_Vel_mps') or 0.0))
            v_list.append(v_val)
        except Exception:
            pass
    max_v = max(v_list) if v_list else 0.0
    avg_v = (sum(v_list) / len(v_list)) if v_list else 0.0

    # Quãng đường
    dist = 0.0
    for r in reversed(rows):
        try:
            d_val = float(r.get('Dist_Traveled_m') or 0.0)
            if d_val > 0.0:
                dist = d_val
                break
        except Exception:
            pass
    if dist <= 0.0:
        # Tích phân sơ bộ từ vận tốc
        dist = avg_v * duration

    # CNN confidence
    conf_list = []
    for r in rows:
        try:
            c_val = float(r.get('Confidence') or 0.0)
            if c_val >= 0.0:
                conf_list.append(c_val)
        except Exception:
            pass
    avg_conf = (sum(conf_list) / len(conf_list) * 100.0) if conf_list else 0.0

    # Trạng thái FSM & Sự kiện
    states = [str(r.get('FSM_State') or '') for r in rows]
    events = [str(r.get('Event') or '') for r in rows]

    has_uturn = any(('UTURN' in s or 'PATH_FOLLOWING' in s) for s in states if s)
    uturn_success = any('UTURN_SUCCESS' in e for e in events if e)
    mission_complete = any('MISSION_COMPLETE' in e for e in events if e)

    if uturn_success:
        uturn_status = 'THÀNH_CÔNG (+180°)'
    elif has_uturn:
        uturn_status = 'ĐANG_QUAY_DỞ'
    else:
        uturn_status = 'TRONG_LUỐNG (CHƯA_QUAY)'

    if mission_complete:
        outcome = '🏆 HOÀN THÀNH 2 LUỐNG'
    elif uturn_success:
        outcome = '✅ ĐÃ SANG LUỐNG 2'
    elif duration > 3.0 and dist > 0.5:
        outcome = '⏹️ DỪNG CHỦ ĐỘNG (CTRL+C)'
    else:
        outcome = '⚠️ DỪNG SỚM (<0.5m)'

    return {
        'Thoi_Gian_Bat_Dau': start_str,
        'Thoi_Luong_s': round(duration, 1),
        'Quang_Duong_m': round(dist, 2),
        'Van_Toc_Max_mps': round(max_v, 3),
        'Van_Toc_TB_mps': round(avg_v, 3),
        'CNN_Confidence_TB': round(avg_conf, 1),
        'Quay_Dau_Omega': uturn_status,
        'Ket_Qua_Nhiem_Vu': outcome,
        'File_Telemetry': filename
    }


def update_history_logbook():
    """Quét toàn bộ logs và tạo sổ nhật ký tổng hợp."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    all_csvs = sorted(glob.glob(os.path.join(LOGS_DIR, "telemetry_*.csv")))
    # Loại bỏ file symlink
    all_csvs = [p for p in all_csvs if not os.path.islink(p) and not p.endswith('latest_telemetry.csv')]

    records = []
    for p in all_csvs:
        info = analyze_single_telemetry(p)
        if info:
            records.append(info)

    # Ghi file CSV
    with open(CSV_HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for idx, rec in enumerate(records, 1):
            writer.writerow([
                idx,
                rec['Thoi_Gian_Bat_Dau'],
                rec['Thoi_Luong_s'],
                rec['Quang_Duong_m'],
                rec['Van_Toc_Max_mps'],
                rec['Van_Toc_TB_mps'],
                rec['CNN_Confidence_TB'],
                rec['Quay_Dau_Omega'],
                rec['Ket_Qua_Nhiem_Vu'],
                rec['File_Telemetry']
            ])

    # Ghi file Markdown đẹp mắt
    with open(MD_HISTORY_FILE, mode='w', encoding='utf-8') as f:
        f.write("# 📋 SỔ NHẬT KÝ THEO DÕI CÁC LẦN CHẠY ROBOT (RUN HISTORY LOGBOOK)\n\n")
        f.write(f"> **Tổng hợp:** {len(records)} lần chạy | Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| STT | Thời Gian Bắt Đầu | Thời Lượng | Quãng Đường | Vận Tốc TB | Tin Cậy AI | Quay Đầu Omega | Kết Quả Nhiệm Vụ | Tệp Telemetry |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |\n")
        for idx, rec in enumerate(records, 1):
            f.write(
                f"| **#{idx:02d}** | {rec['Thoi_Gian_Bat_Dau']} | {rec['Thoi_Luong_s']}s | "
                f"**{rec['Quang_Duong_m']}m** | {rec['Van_Toc_TB_mps']} m/s | "
                f"{rec['CNN_Confidence_TB']}% | {rec['Quay_Dau_Omega']} | "
                f"{rec['Ket_Qua_Nhiem_Vu']} | `{rec['File_Telemetry']}` |\n"
            )

    return records


def print_history_table(limit=15):
    """In bảng nhật ký ra Terminal với định dạng đẹp mắt."""
    records = update_history_logbook()
    if not records:
        print("Chưa có dữ liệu lần chạy nào trong logs/")
        return

    print("\n" + "=" * 105)
    print("                 📋 SỔ NHẬT KÝ THEO DÕI CÁC LẦN CHẠY XE (ROBOT RUN HISTORY)")
    print("=" * 105)
    print(f"{'STT':^5} | {'Thời Gian Bắt Đầu':^19} | {'Thời Lượng':^10} | {'Quãng Đường':^11} | {'V_TB':^8} | {'Tin Cậy AI':^10} | {'Kết Quả Lần Chạy':^25}")
    print("-" * 105)

    displayed = records[-limit:] if limit > 0 else records
    start_idx = len(records) - len(displayed) + 1

    for i, rec in enumerate(displayed, start_idx):
        dur_str = f"{rec['Thoi_Luong_s']}s"
        dist_str = f"{rec['Quang_Duong_m']}m"
        v_str = f"{rec['Van_Toc_TB_mps']}m/s"
        conf_str = f"{rec['CNN_Confidence_TB']}%"
        outcome = rec['Ket_Qua_Nhiem_Vu']
        print(f"#{i:02d}  | {rec['Thoi_Gian_Bat_Dau']:19s} | {dur_str:^10s} | {dist_str:^11s} | {v_str:^8s} | {conf_str:^10s} | {outcome}")

    print("=" * 105)
    print(f"👉 File CSV chi tiết   : logs/lich_su_chay_xe.csv")
    print(f"👉 File Markdown bảng  : logs/lich_su_chay_xe.md")
    print(f"👉 Xem lại chi tiết    : gõ 'xem-lai' (lần mới nhất) hoặc 'plot-lai' (vẽ đồ thị)")
    print("=" * 105 + "\n")


if __name__ == '__main__':
    limit = 20
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        limit = int(sys.argv[1])
    print_history_table(limit=limit)
