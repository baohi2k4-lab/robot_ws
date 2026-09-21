#!/usr/bin/env bash
# ==============================================================================
# Script: sync_to_pi.sh
# Đồng bộ mã nguồn từ PC sang Raspberry Pi và tự động build lại trên Pi
# Cách dùng:
#   ./scripts/sync_to_pi.sh [IP_CUA_PI]
# Ví dụ:
#   ./scripts/sync_to_pi.sh
#   ./scripts/sync_to_pi.sh 10.10.178.200
#   ./scripts/sync_to_pi.sh 192.168.43.10
# ==============================================================================

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WS_DIR="$(dirname "$SCRIPT_DIR")"

PI_USER="bao"
PI_DEFAULT_IP="10.10.178.200"
PI_IP="${1:-$PI_DEFAULT_IP}"

echo "📡 ============================================================"
echo "🚀 ĐỒNG BỘ MÃ NGUỒN PC ➔ RASPBERRY PI"
echo "   IP Pi: $PI_IP (User: $PI_USER)"
echo "   Thư mục PC: $WS_DIR"
echo "📡 ============================================================"

# 1. Kiểm tra kết nối mạng tới Pi
echo "🔍 Đang kiểm tra kết nối mạng tới $PI_IP..."
if ! ping -c 1 -W 2 "$PI_IP" >/dev/null 2>&1; then
    echo "❌ Không thể ping thấy Raspberry Pi tại địa chỉ: $PI_IP"
    echo "⚠️ Nguyên nhân có thể do:"
    echo "   1. Pi đang tắt nguồn hoặc đang khởi động lại."
    echo "   2. Pi và Laptop không cùng mạng Wi-Fi (ví dụ Pi ở mạng 4G/Hotspot còn PC ở CTU)."
    echo "   3. Pi bị đổi IP DHCP. Hãy kiểm tra màn hình Pi hoặc gõ 'hostname -I' trên Pi."
    echo ""
    echo "💡 Gợi ý: Nếu biết IP mới của Pi, hãy chạy:"
    echo "   ./scripts/sync_to_pi.sh <IP_MOI>"
    exit 1
fi
echo "✅ Kết nối mạng OK!"

# 2. Đồng bộ các thư mục và file mã nguồn chính
echo "📦 Đang đồng bộ các package qua rsync..."

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$WS_DIR/src/my_robot_controller/" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/src/my_robot_controller/"

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$WS_DIR/src/my_robot_bringup/" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/src/my_robot_bringup/"

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$WS_DIR/src/my_robot_navigation/" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/src/my_robot_navigation/"

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$WS_DIR/src/my_robot_description/" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/src/my_robot_description/"

rsync -avz --progress \
  "$WS_DIR/src/esp32_source/" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/src/esp32_source/"

rsync -avz --progress \
  "$WS_DIR/aliases.sh" \
  "$WS_DIR/cyclonedds.xml" \
  "${PI_USER}@${PI_IP}:/home/${PI_USER}/robot_ws/"

echo "✅ Đã chuyển toàn bộ tệp tin sang Pi thành công!"

# 3. Đồng bộ thời gian hệ thống
echo "🕒 Đang đồng bộ giờ hệ thống giữa PC và Pi..."
PC_TIME=$(date -u +"%Y-%m-%d %H:%M:%S")
ssh -o StrictHostKeyChecking=no "${PI_USER}@${PI_IP}" "sudo date -u -s '$PC_TIME' >/dev/null 2>&1 || true"
echo "✅ Đã đồng bộ giờ hệ thống!"

# 4. Thực hiện colcon build trên Pi
echo "🔨 Đang tiến hành colcon build trên Raspberry Pi..."
ssh -o StrictHostKeyChecking=no "${PI_USER}@${PI_IP}" "bash -c 'source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null; cd /home/${PI_USER}/robot_ws && colcon build --symlink-install --packages-select my_robot_controller my_robot_bringup my_robot_navigation'"

echo ""
echo "🎉 ============================================================"
echo "✅ ĐỒNG BỘ VÀ BIÊN DỊCH THÀNH CÔNG TRÊN RASPBERRY PI!"
echo "   Bây giờ bạn có thể khởi động xe thật trực tiếp trên Pi."
echo "🎉 ============================================================"
