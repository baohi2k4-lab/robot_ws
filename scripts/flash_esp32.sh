#!/usr/bin/env bash
# ==============================================================================
# Script: flash_esp32.sh
# Nạp trực tiếp Firmware code0409.ino vào vi điều khiển ESP32-S3 qua USB
# Sử dụng: ./scripts/flash_esp32.sh [PORT]
# Ví dụ:   ./scripts/flash_esp32.sh /dev/esp32
# ==============================================================================

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WS_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-/dev/esp32}"

if [ ! -e "$PORT" ]; then
    PORT="/dev/ttyACM0"
fi

if [ ! -e "$PORT" ]; then
    # Kiểm tra xem có đang chạy trên PC và có kết nối Raspberry Pi hay không
    if ping -c 1 -W 1 192.168.10.2 >/dev/null 2>&1; then
        echo "💡 Không tìm thấy $PORT cục bộ trên Laptop."
        echo "📡 Phát hiện Raspberry Pi (192.168.10.2) đang kết nối -> Chuyển tiếp lệnh nạp ESP32 qua Pi..."
        ssh -t bao@192.168.10.2 "nap-esp"
        exit $?
    fi
    echo "❌ Không tìm thấy cổng $PORT! Hãy kiểm tra cáp USB cắm vào ESP32."
    exit 1
fi

echo "============================================================"
echo "⚡ NẠP FIRMWARE ESP32-S3 QUA CỔNG: $PORT"
echo "============================================================"

# Giải phóng cổng nếu có process đang mở
sudo fuser -k "$PORT" 2>/dev/null || true
sleep 0.5

FIRMWARE_DIR="$WS_DIR/esp32_firmware"
if [ ! -f "$FIRMWARE_DIR/code0409.ino.bin" ]; then
    FIRMWARE_DIR="/home/bao/esp32_firmware"
fi

if [ ! -f "$FIRMWARE_DIR/code0409.ino.bin" ]; then
    echo "❌ Không tìm thấy file firmware binary trong $FIRMWARE_DIR!"
    exit 1
fi

echo "🚀 Đang ghi Flash vào ESP32-S3 (Baud: 460800)..."
python3 -m esptool --chip esp32s3 --port "$PORT" --baud 460800 write-flash -z \
    0x00000 "$FIRMWARE_DIR/code0409.ino.bootloader.bin" \
    0x08000 "$FIRMWARE_DIR/code0409.ino.partitions.bin" \
    0x10000 "$FIRMWARE_DIR/code0409.ino.bin"

echo ""
echo "✅ NẠP FIRMWARE ESP32-S3 THÀNH CÔNG RỰC RỠ!"
