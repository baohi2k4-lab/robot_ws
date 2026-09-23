#!/usr/bin/env bash

# Tự động lấy thư mục gốc của Workspace (chạy đúng trên cả PC và Raspberry Pi)
if [ -n "${BASH_SOURCE[0]}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    WS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
fi
if [ -z "$WS_DIR" ] || [ ! -d "$WS_DIR" ]; then
    if [ -d "$HOME/Màn hình nền/robot_ws" ]; then
        WS_DIR="$HOME/Màn hình nền/robot_ws"
    elif [ -d "$HOME/robot-ws" ]; then
        WS_DIR="$(readlink -f "$HOME/robot-ws")"
    elif [ -d "$HOME/robot_ws" ]; then
        WS_DIR="$(readlink -f "$HOME/robot_ws")"
    fi
fi
export WS_DIR

# Hàm nạp môi trường ROS 2 và Workspace
load_ws() {
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi

    # Tự động khắc phục nếu build/my_robot_controller bị mất hook symlink
    if [ -d "$WS_DIR/src/my_robot_controller" ] && [ ! -d "$WS_DIR/build/my_robot_controller" ]; then
        echo "🔧 Phát hiện thiếu thư mục build, đang tự động build nhanh my_robot_controller..."
        (cd "$WS_DIR" && colcon build --symlink-install --packages-select my_robot_controller >/dev/null 2>&1) || true
    fi

    if [ -f "$WS_DIR/install/setup.bash" ]; then
        source "$WS_DIR/install/setup.bash"
    fi


    export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
    export RCUTILS_COLORIZED_OUTPUT=1
    export PYTHONUNBUFFERED=1
    export RCUTILS_LOGGING_BUFFERED_STREAM=0

    export ROS_DOMAIN_ID=42
    export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
    if [ -f "/opt/ros/jazzy/lib/librmw_cyclonedds_cpp.so" ] || [ -f "/opt/ros/humble/lib/librmw_cyclonedds_cpp.so" ] || [ -f "/usr/lib/librmw_cyclonedds_cpp.so" ]; then
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        if [ -f "$WS_DIR/cyclonedds.xml" ]; then
            export CYCLONEDDS_URI="file://$WS_DIR/cyclonedds.xml"
            cp -f "$WS_DIR/cyclonedds.xml" "$HOME/.cyclonedds.xml" 2>/dev/null || true
        elif [ -f "$HOME/.cyclonedds.xml" ]; then
            export CYCLONEDDS_URI="file://$HOME/.cyclonedds.xml"
        fi
    else
        unset RMW_IMPLEMENTATION
    fi
}

# 1. Biên dịch & Cập nhật Workspace
reload_func() {
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
    [ -f "$WS_DIR/aliases.sh" ] && source "$WS_DIR/aliases.sh"
    echo "✅ Đã cập nhật nạp lại toàn bộ phím tắt mới nhất!"
}
alias reload="reload_func"
alias capnhat="reload_func"
alias update-cmd="reload_func"

# Hàm build thông minh (tự động nhận diện distro Jazzy / Humble & tự động nạp phím tắt)
build_func() {
    cd "$WS_DIR"
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi
    colcon build --symlink-install "$@"
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "⚠️ Thử dọn dẹp cache build của my_robot_controller và biên dịch lại..."
        rm -rf "$WS_DIR/build/my_robot_controller"
        colcon build --symlink-install "$@"
        ret=$?
    fi
    if [ $ret -eq 0 ]; then
        if [ -f "$WS_DIR/install/setup.bash" ]; then
            source "$WS_DIR/install/setup.bash"
        fi
        if [ -f "$WS_DIR/aliases.sh" ]; then
            source "$WS_DIR/aliases.sh"
        fi
        echo "✅ [BUILD THÀNH CÔNG] Đã tự động cập nhật & nạp toàn bộ lệnh tắt mới nhất (rviz, quay-rviz, real-robot, wasd,...)!"
    else
        echo "❌ [BUILD THẤT BẠI] Vui lòng kiểm tra lại lỗi code ở trên."
        return $ret
    fi
}
alias build="build_func"
alias rb-build="build_func"

build_all_func() {
    cd "$WS_DIR"
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi
    colcon build --symlink-install "$@"
    local ret=$?
    if [ $ret -eq 0 ]; then
        if [ -f "$WS_DIR/install/setup.bash" ]; then
            source "$WS_DIR/install/setup.bash"
        fi
        if [ -f "$WS_DIR/aliases.sh" ]; then
            source "$WS_DIR/aliases.sh"
        fi
        echo "✅ [BUILD-ALL THÀNH CÔNG] Đã tự động cập nhật & nạp toàn bộ lệnh tắt mới nhất!"
    else
        echo "❌ [BUILD-ALL THẤT BẠI] Vui lòng kiểm tra lại lỗi code ở trên."
        return $ret
    fi
}
alias build-all="build_all_func"

# Lệnh 1-Click đồng bộ nhanh từ GitHub về máy
git_sync_func() {
    cd "$WS_DIR"
    echo "📥 Đang kéo mã nguồn mới nhất từ GitHub..."
    local branch
    branch=$(git branch --show-current 2>/dev/null || echo "main")
    git pull origin "$branch" 2>/dev/null || git pull origin main 2>/dev/null || git pull || true
    rm -rf "$WS_DIR/build/my_robot_controller/models" 2>/dev/null || true
    echo "🔨 Đang biên dịch lại Workspace & cập nhật phím tắt..."
    build_func "$@"
}
alias git-sync="git_sync_func"
alias dongbo="git_sync_func"
alias sync-code="git_sync_func"
alias sync="git_sync_func"

# 2. Các lệnh chạy Mô phỏng (PC)
alias sim="load_ws && ros2 launch my_robot_simulation sim.launch.py"
alias gazebo="load_ws && ros2 launch my_robot_simulation sim.launch.py use_rviz:=false"
alias sim-only="gazebo"
alias ai="load_ws && ros2 launch my_robot_controller control.launch.py"
alias slam="load_ws && ros2 launch my_robot_slam slam.launch.py"
alias nav="load_ws && ros2 launch my_robot_navigation nav.launch.py"

# Bàn phím điều khiển robot (WASD & IJKL - Chạy liên tục ~15 Hz nuôi Watchdog ESP32)
alias teleop="load_ws && ros2 run my_robot_controller teleop_keyboard"
alias ros-teleop="teleop"
alias keyboard="teleop"
alias lai-xe="teleop"
alias wasd="teleop"
alias teleop-wasd="teleop"

# Mở RViz + Camera USB (Hiển thị mô hình xe 3D + Khung hình USB Webcam)
rviz_record_func() {
    load_ws
    mkdir -p "$WS_DIR/dataset/videos" "$WS_DIR/dataset/imgs"
    ros2 launch my_robot_bringup laptop_record.launch.py "$@"
}
alias rviz-record="rviz_record_func"
alias rviz-cam="rviz_record_func"

rviz_view_func() {
    load_ws
    local rviz_file=""
    for cand in \
        "$WS_DIR/src/my_robot_description/rviz/display.rviz" \
        "$HOME/robot-ws/src/my_robot_description/rviz/display.rviz" \
        "$HOME/robot_ws/src/my_robot_description/rviz/display.rviz" \
        "$HOME/Màn hình nền/robot_ws/src/my_robot_description/rviz/display.rviz"; do
        if [ -f "$cand" ]; then
            rviz_file="$cand"
            break
        fi
    done

    export QT_QPA_PLATFORM=xcb
    echo "🖥️ [RVIZ] Đang khởi động giao diện điều khiển 3D..."
    if [ -n "$rviz_file" ]; then
        rviz2 -d "$rviz_file" "$@"
    else
        rviz2 "$@"
    fi
}
alias rviz="rviz_view_func"
alias laptop-view="rviz_view_func"

rviz_only_func() {
    load_ws
    export QT_QPA_PLATFORM=xcb
    local rviz_file=""
    for cand in \
        "$WS_DIR/src/my_robot_description/rviz/display.rviz" \
        "$HOME/robot-ws/src/my_robot_description/rviz/display.rviz" \
        "$HOME/robot_ws/src/my_robot_description/rviz/display.rviz" \
        "$HOME/Màn hình nền/robot_ws/src/my_robot_description/rviz/display.rviz"; do
        if [ -f "$cand" ]; then
            rviz_file="$cand"
            break
        fi
    done

    if [ -n "$rviz_file" ]; then
        rviz2 -d "$rviz_file" "$@"
    else
        rviz2 "$@"
    fi
}
alias rviz-only="rviz_only_func"
alias plot="load_ws && ros2 run my_robot_controller plot_response --mode telemetry"
alias plot-pp="load_ws && ros2 run my_robot_controller plot_response --mode pure_pursuit"
alias plot-smc="load_ws && ros2 run my_robot_controller plot_response --mode smc"
alias plot-gui="load_ws && ros2 run my_robot_controller plot_response --gui"
alias cancel="load_ws && ros2 run my_robot_navigation cancel_nav"

# Hàm lưu bản đồ nhanh
savemap() {
    load_ws
    local map_name="${1:-my_farm_map}"
    mkdir -p "$WS_DIR/maps"
    cd "$WS_DIR/maps" && ros2 run my_robot_slam map_saver -f "$map_name"
    echo "Đã lưu bản đồ '$map_name' vào thư mục: $WS_DIR/maps/"
}

# 3. Các lệnh kiểm tra cảm biến riêng lẻ (1-Click Test)
alias test-lidar="load_ws && ros2 launch my_sensor_test test_lidar.launch.py"
alias test-cam="load_ws && bash \"$WS_DIR/src/my_sensor_test/scripts/run_test_camera.sh\""
alias test-camera="test-cam"
alias test-imu="load_ws && bash \"$WS_DIR/src/my_sensor_test/scripts/run_test_imu.sh\""
alias test-gps="load_ws && ros2 run my_robot_controller gps_driver --ros-args -p serial_port:=/dev/ttyAMA0 -p baudrate:=38400"
alias run-encoder="load_ws && ros2 run encoder_odom encoder_node --ros-args -p serial_port:=/dev/ttyACM0"
alias encoder="run-encoder"
alias test-esp32="load_ws && ros2 run my_robot_bringup esp32_bridge --ros-args -p serial_port:=/dev/esp32"

# Lệnh nạp Firmware ESP32-S3 tự động từ Raspberry Pi
nap_esp32_func() {
    bash "$WS_DIR/src/esp32_source/nap_code_esp32.sh" "$@"
}
alias nap-esp32="nap_esp32_func"
alias flash-esp32="nap_esp32_func"
alias test-all="load_ws && ros2 launch my_sensor_test test_all_sensors.launch.py"
alias test-slam="load_ws && bash \"$WS_DIR/src/my_sensor_test/scripts/run_test_slam.sh\""

# Hàm tự động phát hiện nguồn camera tối ưu (Ưu tiên Camera iPhone DroidCam)
detect_camera_device() {
    for arg in "$@"; do
        if [[ "$arg" == camera_device* ]]; then
            echo "$arg"
            return 0
        fi
    done
    # Ưu tiên 1: Tự động kết nối Camera iPhone (DroidCam qua Wi-Fi Hotspot / Cáp USB 172.20.10.1)
    if ip route 2>/dev/null | grep -q "172.20.10" || ping -c 1 -W 1 172.20.10.1 &>/dev/null; then
        echo "camera_device:=http://172.20.10.1:4747/video"
        return 0
    fi
    # Ưu tiên 2: Cổng USB Webcam vật lý /dev/video0 nếu có
    if [ -e "/dev/video0" ]; then
        echo "camera_device:=/dev/video0"
        return 0
    fi
    # Mặc định: Luồng iPhone DroidCam
    echo "camera_device:=http://172.20.10.1:4747/video"
    return 0
}

# Hàm tự động phát hiện cổng RPLIDAR C1 (/dev/rplidar hoặc fallback /dev/ttyUSB*)
detect_lidar_port() {
    for arg in "$@"; do
        if [[ "$arg" == serial_port* ]]; then
            echo "$arg"
            return 0
        fi
    done
    if [ -e "/dev/rplidar" ]; then
        echo "serial_port:=/dev/rplidar"
        return 0
    fi
    for p in /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3; do
        if [ -e "$p" ]; then
            if [ -e "/dev/esp32" ] && [ "$(readlink -f "$p" 2>/dev/null)" = "$(readlink -f /dev/esp32 2>/dev/null)" ]; then
                continue
            fi
            echo "serial_port:=$p"
            return 0
        fi
    done
    echo "serial_port:=/dev/rplidar"
    return 0
}

# Hàm tự động phát hiện cổng ESP32 (/dev/esp32 hoặc fallback /dev/ttyUSB*)
detect_esp32_port() {
    for arg in "$@"; do
        if [[ "$arg" == esp32_port* ]]; then
            echo "$arg"
            return 0
        fi
    done
    if [ -e "/dev/esp32" ]; then
        echo "esp32_port:=/dev/esp32"
        return 0
    fi
    for p in /dev/ttyUSB1 /dev/ttyUSB0 /dev/ttyUSB2 /dev/ttyUSB3; do
        if [ -e "$p" ]; then
            if [ -e "/dev/rplidar" ] && [ "$(readlink -f "$p" 2>/dev/null)" = "$(readlink -f /dev/rplidar 2>/dev/null)" ]; then
                continue
            fi
            echo "esp32_port:=$p"
            return 0
        fi
    done
    echo "esp32_port:=/dev/esp32"
    return 0
}

# 4. Các lệnh chạy trên Robot Thật (Raspberry Pi)
real_robot_func() {
    load_ws
    # Tự động sửa chữa nếu my_robot_controller bị lỗi package not found
    if ! ros2 pkg prefix my_robot_controller &>/dev/null; then
        echo "🔧 Đang biên dịch lại package my_robot_controller..."
        (cd "$WS_DIR" && colcon build --symlink-install --packages-select my_robot_controller)
        [ -f "$WS_DIR/install/setup.bash" ] && source "$WS_DIR/install/setup.bash"
    fi
    # Giải phóng tiến trình camera hoặc serial kẹt từ lần chạy trước
    fuser -k /dev/video* /dev/ttyUSB* /dev/rplidar /dev/esp32 2>/dev/null || true
    killall -q -9 camera_publisher wifi_cam_bridge sllidar_node rplidar_node esp32_bridge imu_driver costmap_node 2>/dev/null || true

    local cam_arg
    cam_arg=$(detect_camera_device "$@")
    local has_cam=false
    for a in "$@"; do [[ "$a" == camera_device* ]] && has_cam=true; done

    local lidar_port_arg
    lidar_port_arg=$(detect_lidar_port "$@")
    local has_lidar_port=false
    for a in "$@"; do [[ "$a" == serial_port* ]] && has_lidar_port=true; done
    [ "$has_lidar_port" = true ] && lidar_port_arg=""

    local esp32_port_arg
    esp32_port_arg=$(detect_esp32_port "$@")
    local has_esp32_port=false
    for a in "$@"; do [[ "$a" == esp32_port* ]] && has_esp32_port=true; done
    [ "$has_esp32_port" = true ] && esp32_port_arg=""

    if [ "$has_cam" = true ]; then
        ros2 launch my_robot_bringup real_robot.launch.py $lidar_port_arg $esp32_port_arg "$@" 2>&1 | python3 "$WS_DIR/scripts/clean_log_filter.py"
    else
        if [[ "$cam_arg" == *"172.20.10.1"* ]]; then
            echo "📱 Phát hiện iPhone qua cáp USB (172.20.10.1), tự động kích hoạt: $cam_arg"
        fi
        ros2 launch my_robot_bringup real_robot.launch.py $lidar_port_arg $esp32_port_arg "$cam_arg" "$@" 2>&1 | python3 "$WS_DIR/scripts/clean_log_filter.py"
    fi
}
alias real-robot="real_robot_func"
alias real-hw="real-robot enable_camera:=false"
alias xe-that-hw="real-robot enable_camera:=false"
alias hw-only="real-robot enable_camera:=false"
alias real-slam="load_ws && ros2 launch my_robot_bringup real_slam.launch.py"

# Lệnh TỰ HÀNH XE THẬT NGOÀI TRỜI VỚI NAV2 (Odom-Only, không cần bản đồ)
# Trên Pi: chạy real-nav → phần cứng + Nav2 (LiDAR obstacle avoidance)
# Trên PC: mở RViz (Fixed Frame = odom) → Click "2D Nav Goal" → Robot tự đi
unalias real-nav 2>/dev/null || true
real_nav_func() {
    load_ws
    echo "🧭 Khởi động Nav2 Odom-Only (LiDAR obstacle avoidance, không cần bản đồ)"
    echo "📱 Trên PC: mở RViz → Fixed Frame = odom → Click '2D Nav Goal'"
    ros2 launch my_robot_bringup real_nav.launch.py "$@"
}
alias real-nav="real_nav_func"
alias nav-real="real_nav_func"

# Lệnh TỰ HÀNH XE THẬT BÁM LUỐNG BẰNG AI CNN (Crop Row Following)
real_cnn_func() {
    load_ws
    # Tự động sửa chữa nếu my_robot_controller bị lỗi package not found
    if ! ros2 pkg prefix my_robot_controller &>/dev/null; then
        echo "🔧 Đang biên dịch lại package my_robot_controller..."
        (cd "$WS_DIR" && colcon build --symlink-install --packages-select my_robot_controller)
        [ -f "$WS_DIR/install/setup.bash" ] && source "$WS_DIR/install/setup.bash"
    fi
    # Giải phóng tiến trình camera, serial, lidar hoặc node AI kẹt từ lần chạy trước
    fuser -k /dev/video* /dev/ttyUSB* /dev/rplidar /dev/esp32 2>/dev/null || true
    killall -q -9 camera_publisher wifi_cam_bridge cnn_driver sllidar_node rplidar_node costmap_node esp32_bridge imu_driver 2>/dev/null || true

    mkdir -p "$WS_DIR/logs"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local term_log="$WS_DIR/logs/terminal_real_cnn_${timestamp}.log"
    local latest_term="$WS_DIR/logs/terminal_real_cnn_latest.log"

    echo "================================================================================"
    echo "🌾 [REAL-CNN] KHỞI ĐỘNG TỰ HÀNH BÁM LUỐNG BẰNG AI"
    echo "📁 Dữ liệu góc lái & log tự động lưu tại: $WS_DIR/logs/"
    echo "================================================================================"

    local cam_arg
    cam_arg=$(detect_camera_device "$@")
    local has_cam=false
    for a in "$@"; do [[ "$a" == camera_device* ]] && has_cam=true; done

    local gps_arg="enable_gps:=true"
    for a in "$@"; do [[ "$a" == enable_gps* ]] && gps_arg=""; done

    local lidar_arg="enable_lidar:=true"
    for a in "$@"; do [[ "$a" == enable_lidar* ]] && lidar_arg=""; done

    local costmap_arg="enable_costmap:=true"
    for a in "$@"; do [[ "$a" == enable_costmap* ]] && costmap_arg=""; done

    local lidar_port_arg
    lidar_port_arg=$(detect_lidar_port "$@")
    local has_lidar_port=false
    for a in "$@"; do [[ "$a" == serial_port* ]] && has_lidar_port=true; done
    [ "$has_lidar_port" = true ] && lidar_port_arg=""

    local esp32_port_arg
    esp32_port_arg=$(detect_esp32_port "$@")
    local has_esp32_port=false
    for a in "$@"; do [[ "$a" == esp32_port* ]] && has_esp32_port=true; done
    [ "$has_esp32_port" = true ] && esp32_port_arg=""

    echo "📡 [CẢM BIẾN] Đã bật LiDAR C1 ($lidar_port_arg), Costmap ($costmap_arg), ESP32 ($esp32_port_arg)" | tee -a "$term_log"

    export PYTHONUNBUFFERED=1
    export RCUTILS_LOGGING_BUFFERED_STREAM=0
    export RCUTILS_COLORIZED_OUTPUT=1

    if [ "$has_cam" = true ]; then
        ros2 launch my_robot_bringup real_robot.launch.py enable_cnn:=true $gps_arg $lidar_arg $costmap_arg $lidar_port_arg $esp32_port_arg "$@" 2>&1 | python3 "$WS_DIR/scripts/clean_log_filter.py" | tee -a "$term_log"
    else
        if [[ "$cam_arg" == *"172.20.10.1"* ]]; then
            echo "📱 [AI CNN] Tự động kết nối Camera iPhone qua cáp USB: $cam_arg" | tee -a "$term_log"
        fi
        ros2 launch my_robot_bringup real_robot.launch.py enable_cnn:=true $gps_arg $lidar_arg $costmap_arg $lidar_port_arg $esp32_port_arg "$cam_arg" "$@" 2>&1 | python3 "$WS_DIR/scripts/clean_log_filter.py" | tee -a "$term_log"
    fi

    ln -sf "$term_log" "$latest_term" 2>/dev/null || cp -f "$term_log" "$latest_term" 2>/dev/null || true
    echo ""
    echo "================================================================================"
    echo "✅ [REAL-CNN] Đã dừng xe. Toàn bộ dữ liệu lái xe đã lưu tại: $WS_DIR/logs/"
    echo "👉 Xem đồ thị: gõ 'plot-lai' | Xem bảng số liệu: gõ 'xem-lai'"
    echo "================================================================================"
}
alias real-cnn="real_cnn_func"
alias auto-cnn="real-cnn"
alias cnn-auto="real-cnn"
alias fix-cam="sudo fuser -k /dev/video* 2>/dev/null || true; echo '✅ Đã giải phóng cổng Camera USB /dev/video*!'"

# Phím tắt xem và phân tích dữ liệu log lái xe real-cnn
alias xem-lai="python3 \"$WS_DIR/scripts/view_telemetry.py\""
alias xem-log="python3 \"$WS_DIR/scripts/view_telemetry.py\""
alias log-lai="python3 \"$WS_DIR/scripts/view_telemetry.py\""
alias plot-lai="python3 \"$WS_DIR/scripts/view_telemetry.py\" --plot"
alias xem-anh="xdg-open \"$WS_DIR/logs/latest_driving_plot.png\" 2>/dev/null || eog \"$WS_DIR/logs/latest_driving_plot.png\" 2>/dev/null &"
alias xem-plot="xem-anh"
alias xem-sk="python3 \"$WS_DIR/scripts/view_telemetry.py\" --events"
alias tail-log="tail -n 50 -f \"$WS_DIR/logs/terminal_real_cnn_latest.log\""
alias mo-log="ls -lht \"$WS_DIR/logs\""

# Lệnh chạy xe với Camera iPhone (qua cáp USB hoặc Wi-Fi)
test-iphone() {
    load_ws
    local target="${1:-172.20.10.1}"
    if [[ "$target" != http* ]]; then
        target="http://${target}:4747/video"
    fi
    echo "📱 Đang kết nối kiểm tra Camera iPhone tại: $target"
    ros2 run my_robot_bringup camera_publisher --ros-args -p video_device:="$target"
}
alias test-iphone="test-iphone"

robot-iphone() {
    load_ws
    local target="${1:-172.20.10.1}"
    shift 2>/dev/null || true
    if [[ "$target" != http* && "$target" != camera_device* ]]; then
        target="http://${target}:4747/video"
    fi
    if [[ "$target" == camera_device* ]]; then
        real-robot "$target" "$@"
    else
        real-robot camera_device:="$target" "$@"
    fi
}
alias robot-iphone="robot-iphone"

cnn-iphone() {
    load_ws
    local target="${1:-172.20.10.1}"
    shift 2>/dev/null || true
    if [[ "$target" != http* && "$target" != camera_device* ]]; then
        target="http://${target}:4747/video"
    fi
    if [[ "$target" == camera_device* ]]; then
        real-cnn "$target" "$@"
    else
        real-cnn camera_device:="$target" "$@"
    fi
}
alias cnn-iphone="cnn-iphone"

# Lệnh GÁNH TẢI AI CNN TRÊN LAPTOP (Offloading: Pi giữ nguyên phần cứng & quyết định, Laptop gánh ONNX)
# - Pi (Xe thật): Giữ nguyên camera cắm cáp vào Pi, chạy: real-cnn enable_gps:=true
# - Laptop: Mở terminal chạy: laptop-cnn (Laptop gánh ONNX 30-60 FPS, Pi ra quyết định lái)
laptop_cnn_func() {
    load_ws
    echo "💻 [LAPTOP AI WORKER] Khởi chạy Server tính toán AI CNN trên Laptop..."
    echo "   📡 Nhận ảnh từ Pi (/camera/compressed) ➔ Chạy mạng ONNX (30-60 FPS) ➔ Trả kết quả về Pi (/crop_row/detection)"
    ros2 run my_robot_controller cnn_server "$@"
}
alias laptop-cnn="laptop_cnn_func"
alias cnn-laptop="laptop_cnn_func"
alias offload-cnn="laptop_cnn_func"

# Lệnh CHẠY CAMERA CẮM TRỰC TIẾP VÀO LAPTOP + AI CNN ĐIỀU KHIỂN XE THẬT
# - Pi (Xe thật): Chạy 'real-hw' (chỉ bật phần cứng ESP32 + LiDAR + IMU + GPS + EKF, không bật camera)
# - Laptop: Cắm cáp RealSense D435 hoặc iPhone vào cổng USB Laptop, chạy: 'laptop-ai'
laptop_ai_func() {
    load_ws
    echo "================================================================================"
    echo "💻 [LAPTOP-AI] KHỞI ĐỘNG CAMERA USB TRÊN LAPTOP + AI CNN ĐIỀU KHIỂN XE THẬT"
    echo "   📷 Camera cắm trực tiếp cổng USB Laptop (0ms trễ)"
    echo "   ⚡ Model ONNX chạy GPU/CPU Laptop (30 - 60 FPS)"
    echo "   📡 Nhận LiDAR/IMU/GPS từ Pi ➔ Bắn lệnh /cmd_vel sang Pi lái 4 bánh"
    echo "================================================================================"
    ros2 launch my_robot_bringup laptop_ai.launch.py "$@"
}
alias laptop-ai="laptop_ai_func"
alias xe-laptop="laptop_ai_func"
alias ai-laptop="laptop_ai_func"
alias lap-ai="laptop_ai_func"

# Lệnh KIỂM TRA CHẨN ĐOÁN TOÀN DIỆN CHUỖI AI CNN
alias check-cnn="load_ws && python3 \"$WS_DIR/scripts/verify_cnn_pipeline.py\""

# Lệnh CHẠY THỬ SUY LUẬN AI CNN TRÊN ẢNH (Mô phỏng 100% xe thật bám luống)
alias test-img="load_ws && python3 \"$WS_DIR/scripts/test_inference_images.py\""
alias cnn-img="test-img"

alias pc-nav="load_ws && ros2 launch my_robot_bringup pc_nav.launch.py"
alias nav-slam="load_ws && ros2 launch my_robot_bringup pc_nav.launch.py"
alias gps-nav="load_ws && ros2 launch my_robot_navigation gps_nav.launch.py"
alias nav-gps="gps-nav"

# Lệnh MỞ RVIZ TRÊN PC với đầy đủ Costmap + LiDAR + Nav2 Goal (Click chuột chọn điểm đích)
alias pc-rviz="load_ws && rviz2 -d \"\$(ros2 pkg prefix my_robot_bringup)/share/my_robot_bringup/rviz/nav2_outdoor.rviz\""

# Lệnh GỬI TOẠ ĐỘ ĐÍCH TEST TỰ HÀNH (Chạy trên Laptop hoặc Pi, không cần click chuột)
send_nav_goal() {
    load_ws
    local x="${1:-1.0}"
    local y="${2:-0.0}"
    echo "🎯 Đang gửi lệnh tự hành đến toạ độ (X=$x m, Y=$y m) qua /goal_pose..."
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'odom'}, pose: {position: {x: $x, y: $y, z: 0.0}, orientation: {w: 1.0}}}"
}
alias send-goal="send_nav_goal"
alias test-goal="send_nav_goal"
alias chay-xe="ros2 topic pub --once /start_mission std_msgs/msg/Empty '{}'; echo '🚀 Đã gửi lệnh xuất phát xe!'"
alias bat-xe="chay-xe"

# Lệnh kích hoạt xe THẬT CÓ QUAY VIDEO THÔ (100% Raw, không hiện gì trên màn hình)
real-record() {
    load_ws
    local vname="${1:-}"
    if [ -n "$vname" ]; then
        ros2 launch my_robot_bringup real_robot.launch.py record:=true record_name:="$vname"
    else
        ros2 launch my_robot_bringup real_robot.launch.py record:=true
    fi
}

# Lệnh MỞ RVIZ + XEM XE 3D + WEBCAM USB DVD20 (1080p Full HD @ 60 FPS) + QUAY VIDEO TÁCH FRAME DATASET RIÊNG (Chạy trên Laptop)
rviz-record() {
    load_ws
    local vname="${1:-}"
    if [ -n "$vname" ]; then
        shift
        mkdir -p "$WS_DIR/dataset/videos" "$WS_DIR/dataset/imgs"
        ros2 launch my_robot_bringup laptop_record.launch.py name:="$vname" "$@"
    else
        mkdir -p "$WS_DIR/dataset/videos" "$WS_DIR/dataset/imgs"
        ros2 launch my_robot_bringup laptop_record.launch.py "$@"
    fi
}
alias record-rviz="rviz-record"
alias laptop-record="rviz-record"
alias quay-rviz="rviz-record"
alias quay="rviz-record"
alias quay-video="rviz-record"

# =====================================================
# TÌM PI & CHUYỂN MẠNG (Hotspot / Mạng trường)
# =====================================================

# Nạp file cấu hình mạng riêng biệt nếu tồn tại
if [ -f "$WS_DIR/network_config.env" ]; then
    # shellcheck source=/dev/null
    source "$WS_DIR/network_config.env"
fi

# Giá trị mặc định dự phòng (nếu chưa điền trong network_config.env)
WIFI_TRUONG="${WIFI_TRUONG:-CTU}"
WIFI_HOTSPOT="${WIFI_HOTSPOT:-VinhHotspot}"
WIFI_HOTSPOT_PASS="${WIFI_HOTSPOT_PASS:-12345678}"
PI_USER="${PI_USER:-bao}"
PI_HOSTNAME="${PI_HOSTNAME:-bao}"
PI_STATIC_IP="${PI_STATIC_IP:-10.10.178.200}"

# Lệnh mở nhanh file cấu hình mạng để chỉnh sửa
edit_net_func() {
    local cfg="$WS_DIR/network_config.env"
    if [ ! -f "$cfg" ]; then
        echo "⚠️ Không tìm thấy file $cfg, đang tạo mới..."
        cat > "$cfg" << 'ENVEOF'
# ==============================================================================
# 📡 CẤU HÌNH THÔNG TIN MẠNG (HOTSPOT ĐIỆN THOẠI & MẠNG TRƯỜNG)
# ==============================================================================
WIFI_TRUONG="CTU"
WIFI_HOTSPOT="VinhHotspot"
WIFI_HOTSPOT_PASS="12345678"
PI_USER="bao"
PI_HOSTNAME="bao"
PI_STATIC_IP="10.10.178.200"
ENVEOF
    fi

    if command -v nano &>/dev/null; then
        nano "$cfg"
    elif command -v gedit &>/dev/null; then
        gedit "$cfg" &
    else
        echo "📝 Hãy mở file sau để chỉnh sửa:"
        echo "   $cfg"
    fi
    # Tự động nạp lại sau khi đóng trình sửa
    if [ -f "$cfg" ]; then
        source "$cfg"
        echo "✅ Đã nạp cấu hình mạng mới: Hotspot='$WIFI_HOTSPOT', Trường='$WIFI_TRUONG', User='$PI_USER'"
    fi
}
alias edit-net="edit_net_func"
alias cauhinh-mang="edit_net_func"
alias sua-mang="edit_net_func"

# Hàm tìm IP của Raspberry Pi trên bất kỳ mạng nào (Chạy trên PC)
find_pi_func() {
    local found_ip=""

    # Cách 1: Thử mDNS (avahi) — nhanh nhất
    if command -v avahi-resolve &>/dev/null; then
        found_ip=$(avahi-resolve -4 -n "${PI_HOSTNAME}.local" 2>/dev/null | awk '{print $2}')
    fi
    if [ -z "$found_ip" ]; then
        found_ip=$(getent ahosts "${PI_HOSTNAME}.local" 2>/dev/null | awk 'NR==1{print $1}')
    fi

    # Cách 2: Thử ping các IP đã biết (mạng trường)
    if [ -z "$found_ip" ] && [ -n "$PI_STATIC_IP" ]; then
        for ip in $PI_STATIC_IP; do
            if ping -c 1 -W 1 "$ip" &>/dev/null; then
                found_ip="$ip"
                break
            fi
        done
    fi

    # Cách 3: Quét ARP table (tìm trên cùng subnet hiện tại)
    if [ -z "$found_ip" ]; then
        # Lấy subnet hiện tại
        local my_ip my_subnet iface
        iface=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
        if [ -z "$iface" ]; then
            iface=$(ip -4 route show default 2>/dev/null | awk '{print $5}' | head -1)
        fi
        my_ip=$(ip -4 addr show dev "$iface" 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -1)
        if [ -n "$my_ip" ]; then
            my_subnet=$(echo "$my_ip" | sed 's/\.[0-9]*$/.0\/24/')
            # Quét nhanh subnet bằng ping sweep + kiểm tra SSH port 22
            echo "🔍 Đang quét mạng $my_subnet tìm Pi..." >&2
            # Ping sweep nền (kích ARP cache)
            for i in $(seq 1 254); do
                ping -c 1 -W 0.3 "$(echo "$my_subnet" | sed "s|\.0/24|.$i|")" &>/dev/null &
            done
            wait 2>/dev/null
            sleep 1
            # Kiểm tra ARP table, thử SSH vào từng IP tìm Pi
            while IFS= read -r candidate; do
                if [ -n "$candidate" ] && [ "$candidate" != "$my_ip" ]; then
                    # Thử kết nối SSH nhanh (timeout 2s)
                    if timeout 2 bash -c "echo >/dev/tcp/$candidate/22" 2>/dev/null; then
                        # Xác nhận đây là Pi bằng hostname
                        local remote_host
                        remote_host=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 -o BatchMode=yes "${PI_USER}@${candidate}" "hostname" 2>/dev/null)
                        if [ "$remote_host" = "$PI_HOSTNAME" ]; then
                            found_ip="$candidate"
                            break
                        fi
                    fi
                fi
            done < <(arp -n 2>/dev/null | awk '/ether/{print $1}')
        fi
    fi

    if [ -n "$found_ip" ]; then
        echo "$found_ip"
    else
        echo "" >&2
        echo "❌ Không tìm thấy Raspberry Pi ($PI_HOSTNAME) trên mạng hiện tại!" >&2
        echo "   Hãy kiểm tra Pi đã kết nối cùng mạng Wi-Fi chưa." >&2
        echo "   Hoặc chỉ định IP thủ công: find-pi 192.168.x.y" >&2
        return 1
    fi
}

# Alias find-pi: hiển thị IP Pi và hướng dẫn SSH
find_pi_show() {
    if [ -n "${1:-}" ]; then
        # Người dùng truyền IP thủ công → kiểm tra nhanh
        if ping -c 1 -W 2 "$1" &>/dev/null; then
            echo "✅ Pi tìm thấy tại: $1"
            echo "   👉 SSH:  ssh ${PI_USER}@${1}"
            export PI_IP="$1"
        else
            echo "❌ Không ping được $1"
        fi
        return
    fi

    echo "🔍 Đang tìm Raspberry Pi trên mạng..."
    local ip
    ip=$(find_pi_func)
    if [ -n "$ip" ]; then
        echo "✅ Tìm thấy Pi tại: $ip"
        echo "   👉 SSH:  ssh ${PI_USER}@${ip}"
        export PI_IP="$ip"
    fi
}
alias find-pi="find_pi_show"
alias tim-pi="find-pi"

# Hàm đồng bộ thời gian từ Laptop sang Pi qua SSH (Chạy trên Laptop)
sync_time_func() {
    local pi_ip="${1:-${PI_IP:-}}"
    if [ -z "$pi_ip" ]; then
        pi_ip=$(find_pi_func 2>/dev/null)
    fi
    if [ -z "$pi_ip" ]; then
        pi_ip="10.10.178.200" # IP Raspberry Pi hiện tại
    fi
    echo "⏱️ Đang đồng bộ thời gian từ Laptop sang Pi ($pi_ip)..."
    local now_epoch
    now_epoch=$(date +%s)
    ssh -t -o ConnectTimeout=5 "${PI_USER:-bao}@${pi_ip}" "sudo date -s @${now_epoch} && sudo hwclock -w 2>/dev/null || true; echo '✅ Giờ trên Pi hiện tại: '\$(date '+%Y-%m-%d %H:%M:%S')"
}
alias sync-time="sync_time_func"
alias dongbo-gio="sync_time_func"

# Hàm đồng bộ toàn bộ mã nguồn sang Raspberry Pi và tự động build
sync_to_pi_func() {
    local pi_ip="${1:-${PI_STATIC_IP:-}}"
    "$WS_DIR/scripts/sync_to_pi.sh" "$pi_ip"
}
alias sync-pi="sync_to_pi_func"
alias dongbo-pi="sync_to_pi_func"

# Hàm cập nhật CycloneDDS cho unicast (khi hotspot chặn multicast)
update_cyclone_peers() {
    local pi_ip="${1:-}"
    local my_ip
    my_ip=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+')
    if [ -z "$pi_ip" ]; then
        pi_ip=$(find_pi_func)
    fi
    if [ -z "$pi_ip" ] || [ -z "$my_ip" ]; then
        echo "❌ Không xác định được IP Pi hoặc PC để cập nhật CycloneDDS"
        return 1
    fi

    cat > "$WS_DIR/cyclonedds.xml" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface address="${my_ip}" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="${pi_ip}"/>
        <Peer address="${my_ip}"/>
        <Peer address="192.168.10.2"/>
        <Peer address="192.168.10.1"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
XMLEOF
    cp -f "$WS_DIR/cyclonedds.xml" "$HOME/.cyclonedds.xml" 2>/dev/null || true
    echo "✅ Đã cập nhật cyclonedds.xml: Pi=$pi_ip, PC=$my_ip (đã gán đúng IP card mạng)"
}

# Khôi phục CycloneDDS về multicast (dùng trên mạng trường)
restore_cyclone_multicast() {
    cat > "$WS_DIR/cyclonedds.xml" <<'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <General>
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
XMLEOF
    echo "✅ Đã khôi phục cyclonedds.xml về multicast (mạng trường)"
}

# =====================================================
# LỆNH CHUYỂN MẠNG CHO PI (chạy trên Pi qua SSH)
# =====================================================
# Chuyển Pi sang Hotspot điện thoại
switch_hotspot_func() {
    local ssid="${1:-$WIFI_HOTSPOT}"
    local pass="${2:-$WIFI_HOTSPOT_PASS}"
    echo "📱 Đang chuyển Pi sang hotspot: $ssid ..."
    sudo nmcli device wifi connect "$ssid" password "$pass"
    sleep 2
    local new_ip
    new_ip=$(hostname -I | awk '{print $1}')
    echo "✅ Pi đã kết nối hotspot '$ssid'"
    echo "   📍 IP mới của Pi: $new_ip"
    echo "   👉 Trên PC hãy chạy: ssh ${PI_USER}@${new_ip}"
    echo "   👉 Hoặc chạy: find-pi"
}
alias switch-hotspot="switch_hotspot_func"
alias hotspot="switch_hotspot_func"
alias doi-mang="switch_hotspot_func"

# Chuyển Pi về mạng trường
switch_school_func() {
    local ssid="${1:-$WIFI_TRUONG}"
    echo "🏫 Đang chuyển Pi về mạng trường: $ssid ..."
    sudo nmcli device wifi connect "$ssid"
    sleep 2
    local new_ip
    new_ip=$(hostname -I | awk '{print $1}')
    echo "✅ Pi đã kết nối lại mạng trường '$ssid'"
    echo "   📍 IP mới của Pi: $new_ip"
    echo "   👉 Trên PC hãy chạy: ssh ${PI_USER}@${new_ip}"
}
alias switch-school="switch_school_func"
alias mang-truong="switch_school_func"
alias ve-truong="switch_school_func"

# Xem mạng hiện tại
alias check-wifi="nmcli -t -f NAME,DEVICE connection show --active | grep wifi && echo '📍 IP:' && hostname -I"
alias xem-mang="check-wifi"

# =====================================================
# KẾT NỐI TRỰC TIẾP LAPTOP ↔ PI QUA CÁP ETHERNET
# (Độ trễ < 1ms, tốt nhất cho điều khiển AI real-time)
# =====================================================
setup_eth_direct_func() {
    # Tự động tìm cổng Ethernet vật lý (bỏ qua lo, wlo, wlan)
    local eth_iface
    eth_iface=$(ip link show | grep -v 'lo\|wlo\|wlan\|LOOPBACK' | grep -oP '^\d+: \K[^:@]+' | head -1)
    if [ -z "$eth_iface" ]; then
        echo "❌ Không tìm thấy cổng Ethernet! Hãy cắm USB-to-Ethernet adapter trước."
        echo "   Sau khi cắm, chạy lại: setup-eth-direct"
        echo "   Xem interface mới: ip link show"
        return 1
    fi
    local LAPTOP_IP="192.168.10.1"
    local PI_IP_ETH="192.168.10.2"
    echo "🔌 [ETH DIRECT] Cổng tìm thấy: $eth_iface"
    echo "   Laptop IP: $LAPTOP_IP | Pi IP: $PI_IP_ETH"
    # Gán IP tĩnh
    sudo ip addr flush dev "$eth_iface" 2>/dev/null || true
    sudo ip addr add "${LAPTOP_IP}/24" dev "$eth_iface"
    sudo ip link set "$eth_iface" up
    # Cập nhật CycloneDDS dùng unicast IP mới
    cat > "$WS_DIR/cyclonedds.xml" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <General>
      <AllowMulticast>false</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="${PI_IP_ETH}"/>
        <Peer address="${LAPTOP_IP}"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
XMLEOF
    export CYCLONEDDS_URI="file://$WS_DIR/cyclonedds.xml"
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=42
    echo ""
    echo "✅ [ETH DIRECT] Cấu hình xong!"
    echo "   📡 ROS 2 sẽ dùng: $eth_iface (unicast, không multicast)"
    echo "   ⚡ Độ trễ kỳ vọng: < 1ms"
    echo ""
    echo "👉 Bước tiếp theo — Trên Pi chạy:"
    echo "   sudo ip addr add ${PI_IP_ETH}/24 dev eth0"
    echo "   sudo ip link set eth0 up"
    echo "   export ROS_DOMAIN_ID=42"
    echo "   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
    echo ""
    echo "👉 Test kết nối:"
    echo "   ping ${PI_IP_ETH}"
}
alias setup-eth-direct="setup_eth_direct_func"
alias ket-noi-cap="setup_eth_direct_func"
alias eth-direct="setup_eth_direct_func"

# Lệnh tải video từ Pi xuống Laptop (Chạy trên Laptop) — tự động tìm Pi
get-video() {
    local pi_ip="${1:-}"
    if [ -z "$pi_ip" ]; then
        echo "🔍 Đang tìm Pi trên mạng..."
        pi_ip=$(find_pi_func 2>/dev/null)
    fi
    if [ -z "$pi_ip" ]; then
        echo "❌ Không tìm thấy Pi! Hãy truyền IP thủ công: get-video 192.168.x.y"
        return 1
    fi
    mkdir -p "$WS_DIR/dataset"
    echo "📥 Đang kéo video thô từ Pi ($pi_ip) về $WS_DIR/dataset/ ..."
    rsync -avP --include='*.mp4' "${PI_USER}@$pi_ip:~/robot-ws/recordings/" "$WS_DIR/dataset/" 2>/dev/null || \
    rsync -avP --include='*.mp4' "${PI_USER}@$pi_ip:~/robot_ws/recordings/" "$WS_DIR/dataset/" 2>/dev/null || \
    scp "${PI_USER}@$pi_ip:~/robot-ws/recordings/*.mp4" "$WS_DIR/dataset/" 2>/dev/null || \
    scp "${PI_USER}@$pi_ip:~/robot_ws/recordings/*.mp4" "$WS_DIR/dataset/"
    echo "✅ File video đã được lưu tại: $WS_DIR/dataset/"
}
alias get-videos="get-video"

# Lệnh tải toàn bộ Log & CSV lái xe từ Pi về Laptop (Chạy trên Laptop)
get-log() {
    local pi_ip="${1:-}"
    if [ -z "$pi_ip" ]; then
        echo "🔍 Đang tìm Pi trên mạng..."
        pi_ip=$(find_pi_func 2>/dev/null)
    fi
    if [ -z "$pi_ip" ]; then
        echo "❌ Không tìm thấy Pi! Hãy truyền IP thủ công: get-log 192.168.x.y"
        return 1
    fi
    mkdir -p "$WS_DIR/logs"
    echo "📥 Đang kéo toàn bộ Log & CSV từ Pi ($pi_ip) về $WS_DIR/logs/ ..."
    rsync -avP "${PI_USER}@$pi_ip:~/robot-ws/logs/" "$WS_DIR/logs/" 2>/dev/null || \
    rsync -avP "${PI_USER}@$pi_ip:~/robot_ws/logs/" "$WS_DIR/logs/" 2>/dev/null || \
    scp "${PI_USER}@$pi_ip:~/robot-ws/logs/*.*" "$WS_DIR/logs/" 2>/dev/null || \
    scp "${PI_USER}@$pi_ip:~/robot_ws/logs/*.*" "$WS_DIR/logs/"
    echo "✅ Đã đồng bộ toàn bộ log về Laptop tại: $WS_DIR/logs/"
    echo "👉 Bạn có thể gõ ngay 'xem-lai' hoặc 'plot-lai' để phân tích số liệu trên Laptop!"
}
alias get-logs="get-log"
alias pull-log="get-log"
alias sync-log="get-log"

alias extract-dataset="python3 \"$WS_DIR/scripts/extract_dataset.py\""
alias rename-dataset="python3 \"$WS_DIR/scripts/rename_dataset.py\""

# Lệnh xóa toàn bộ video đã quay (chạy được trên cả Pi và Laptop)
clean-video() {
    echo "🗑️ Đang dọn dẹp các video trong thư mục recordings và dataset..."
    rm -f ~/robot-ws/recordings/*.mp4 ~/robot_ws/recordings/*.mp4 "$WS_DIR/recordings/"*.mp4 2>/dev/null || true
    echo "✅ Đã dọn dẹp video hoàn tất!"
}
alias clear-video="clean-video"
alias del-video="clean-video"

# Lệnh xem video nhanh bằng Firefox
play-video() {
    local f="${1:-}"
    if [ -z "$f" ]; then
        f=$(ls -t "$WS_DIR/dataset/videos/"*.mp4 "$WS_DIR/dataset/"*.mp4 2>/dev/null | head -n 1)
    fi
    if [ -n "$f" ] && [ -f "$f" ]; then
        echo "🎬 Đang mở video: $f"
        firefox "$f" &>/dev/null &
    else
        echo "❌ Không tìm thấy file video nào trong $WS_DIR/dataset/videos/"
    fi
}
# =====================================================
# THEO DÕI BỘ LỌC CẢM BIẾN & SENSOR FUSION (IMU, ENCODER, EKF)
# =====================================================

# 1. Xem dữ liệu IMU qua bộ lọc Madgwick (/imu/data)
xem-imu() {
    echo "🧭 [IMU Madgwick] Đang theo dõi hướng và độ nghiêng cảm biến... (Ctrl+C để dừng)"
    python3 -c '
import rclpy, math
from sensor_msgs.msg import Imu

def cb(msg):
    q = msg.orientation
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2 * (q.w * q.y - q.z * q.x)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    wz = msg.angular_velocity.z
    print(f"\r🧭 [IMU Madgwick] Hướng xe (Yaw): {yaw:+6.1f}° | Nghiêng ngang: {roll:+5.1f}° | Dốc trước/sau: {pitch:+5.1f}° | Tốc độ xoay: {wz:+5.2f}rad/s", end="", flush=True)

rclpy.init()
node = rclpy.create_node("xem_imu_cli")
node.create_subscription(Imu, "/imu/data", cb, 10)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    print()
'
}
alias show-imu="xem-imu"
alias check-imu="xem-imu"

# 2. Xem dữ liệu Encoder bánh xe từ ESP32 (/odom/raw)
xem-encoder() {
    echo "🚗 [Encoder ESP32] Đang theo dõi vận tốc và quãng đường lăn bánh... (Ctrl+C để dừng)"
    python3 -c '
import rclpy, math
from nav_msgs.msg import Odometry

def cb(msg):
    vx = msg.twist.twist.linear.x
    wz = msg.twist.twist.angular.z
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    print(f"\r🚗 [Encoder ESP32] Vận tốc lăn: {vx:+5.2f}m/s | Bẻ lái bánh: {wz:+5.2f}rad/s | Quãng đường đã lăn: ({x:+5.2f}, {y:+5.2f})m", end="", flush=True)

rclpy.init()
node = rclpy.create_node("xem_enc_cli")
node.create_subscription(Odometry, "/odom/raw", cb, 10)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    print()
'
}
alias xem-enc="xem-encoder"
alias show-enc="xem-encoder"
alias show-encoder="xem-encoder"

# 3. Xem kết quả dung hợp EKF cuối cùng (/odometry/filtered)
xem-ekf() {
    echo "⭐ [EKF Fusion] Đang theo dõi tọa độ & hướng dung hợp tổng hợp... (Ctrl+C để dừng)"
    python3 -c '
import rclpy, math
from nav_msgs.msg import Odometry

def cb(msg):
    q = msg.pose.pose.orientation
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    vx = msg.twist.twist.linear.x
    wz = msg.twist.twist.angular.z
    print(f"\r⭐ [EKF Tổng Hợp] Tọa độ: (X:{x:+5.2f}m, Y:{y:+5.2f}m) | Hướng chuẩn: {yaw:+6.1f}° | Tốc độ thực: {vx:+5.2f}m/s | Bẻ lái: {wz:+5.2f}rad/s", end="", flush=True)

rclpy.init()
node = rclpy.create_node("xem_ekf_cli")
node.create_subscription(Odometry, "/odometry/filtered", cb, 10)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    print()
'
}
alias show-ekf="xem-ekf"
alias xem-odom="xem-ekf"
alias show-odom="xem-ekf"
alias check-ekf="load_ws && python3 \"$WS_DIR/src/my_sensor_test/scripts/check_ekf\""
alias xem-fusion="check-ekf"
alias test-ekf="check-ekf"
alias check-imu="load_ws && python3 \"$WS_DIR/src/my_sensor_test/scripts/check_imu\""
alias dual-ekf="load_ws && ros2 launch my_robot_bringup dual_ekf.launch.py"
alias ekf-gps="load_ws && ros2 launch my_robot_bringup dual_ekf.launch.py enable_gps:=true"

# =====================================================
# BẢNG TRA CỨU LỆNH TẮT NHANH (ros-help)
# =====================================================
ros_help_func() {
cat << 'EOF'
================================================================================
  🤖 BẢNG TRA CỨU TOÀN BỘ LỆNH TẮT NHANH (ROS 2 ROBOT CHEAT SHEET)
================================================================================

📊 [THEO DÕI BỘ LỌC & SENSOR FUSION] (Mới nhất)
  xem-imu (show-imu) : Xem IMU qua Bộ lọc Madgwick (Roll, Pitch, Yaw theo Độ °)
  xem-enc (xem-encoder): Xem Odometry bánh xe thô từ ESP32 (Vận tốc m/s, Xoay bánh)
  xem-ekf (xem-odom) : Xem KẾT QUẢ DUNG HỢP EKF CUỐI CÙNG (Tọa độ X/Y, Hướng Yaw, Tốc độ)

💻 [TRÊN LAPTOP] (Màn hình quan sát, Lái xe & Xử lý Dataset)
  quay-rviz [tên]    : Mở RViz + Quay video Full HD 1080p 60FPS + Tách Dataset ảnh
                       (Tên khác: rviz-record, laptop-record)
  laptop-cnn [view:=1]: BẬT MODEL AI CNN TRÊN LAPTOP (Cắm cáp sạc điện thoại vào Laptop, xử lý cực nhanh 30-60 FPS)
  laptop-view        : Mở RViz2 nhận luồng Camera nén từ Pi qua Wi-Fi (mượt, không lag)
  teleop (lai-xe)    : Bàn phím lái xe chuẩn gốc ROS 2 (i=tiến, ,=lùi, j/l=rẽ, k=dừng)
  get-video          : Tự động tìm Pi và kéo video MP4 mới quay về máy tính
  play-video (xem)   : Xem ngay video vừa quay bằng trình duyệt Firefox
  clean-video        : Dọn dẹp các video cũ giải phóng ổ đĩa
  extract-dataset <f>: Cắt video thành bộ ảnh sạch (JPG) để gán nhãn train CNN
  rviz               : Mở giao diện RViz2 đồ họa thuần túy
  rviz-only          : Mở giao diện RViz2 cấu hình chuẩn cho xe thật
  pc-nav (nav-slam)  : Bật Nav2 trên Laptop kết hợp với SLAM trực tiếp từ Pi
  cancel             : Hủy mục tiêu dẫn đường Nav2

🍓 [TRÊN RASPBERRY PI] (Khởi động phần cứng xe & Quay video)
  real-robot         : BẬT XE THẬT (Tự động chạy Madgwick + EKF chuẩn xác)
  real-record [tên]  : BẬT XE THẬT + QUAY VIDEO THÔ (100% Raw, lưu MP4 vào Pi)
  real-cnn           : BẬT XE THẬT TỰ HÀNH BÁM LUỐNG BẰNG AI CNN (Tự động lưu toàn bộ log & góc lái)
  chay-xe (bat-xe)   : Gửi lệnh xuất phát cho xe chạy (nếu không bấm 2D Goal Pose trên RViz)
  xem-lai (xem-log)  : Xem báo cáo phân tích góc lái, tốc độ, độ tin cậy AI của lần chạy gần nhất
  tail-log           : Xem luồng log terminal thời gian thực của real-cnn
  mo-log             : Liệt kê toàn bộ các file log CSV & Text trong robot_ws/logs/
  real-slam          : Bật Xe Thật + SLAM Toolbox vẽ bản đồ
  real-nav            : Bật Xe Thật + Nav2 ngoài trời (Odom-Only, LiDAR obstacle avoidance)
  savemap <tên_map>  : Lưu bản đồ SLAM vừa quét xong vào thư mục maps/

🔍 [KIỂM TRA CẢM BIẾN & AI] (1-Click Test trên Pi / Laptop)
  check-cnn (test-cnn): Kiểm tra chẩn đoán toàn diện chuỗi AI CNN (512x512, góc lái, ESP32)
  check-ekf (test-ekf): Chẩn đoán bảng số liệu trực tiếp EKF (Wheel, IMU, GPS, Độ lệch)
  check-imu          : Kiểm tra lọc rung & bù trôi tĩnh ZUPT của cảm biến IMU
  test-cam           : Kiểm tra hình ảnh Webcam DVD20 1080p 60FPS (/dev/video0)
  test-lidar         : Kiểm tra tia quét mắt LiDAR RPLIDAR C1 (/dev/ttyUSB0)
  test-imu           : Kiểm tra cảm biến IMU 9 trục ICM-20948 (I2C Pin 3, 5)
  test-esp32         : Kiểm tra kết nối mạch điều khiển ESP32 Bridge
  nap-esp32          : Nạp firmware code0409.ino vào ESP32-S3 (/dev/esp32)
  run-encoder        : Chạy Node tính toán Odometry từ xung RAW 4 bánh
  test-gps           : Kiểm tra module GPS UART GPIO (/dev/ttyAMA0)
  gps-nav            : Tự hành theo cọc tiêu tọa độ GPS ngoài trời
  test-all           : Kiểm tra toàn bộ cảm biến cùng lúc trên RViz

🎮 [MÔ PHỎNG & ĐỒ THỊ] (Chạy trên PC / Laptop)
  sim                : Mở thế giới ảo Gazebo + RViz + Xe mô phỏng
  gazebo (sim-only)  : Mở riêng Gazebo (không mở RViz)
  ai                 : Bật thuật toán AI CNN bám luống trong mô phỏng
  slam               : Bật SLAM vẽ bản đồ ảo
  nav                : Bật Nav2 dẫn đường trong mô phỏng
  plot               : Vẽ biểu đồ quỹ đạo & cảm biến Telemetry
  plot-pp            : Vẽ phân tích đáp ứng Pure Pursuit
  plot-smc           : Vẽ phân tích bộ điều khiển trượt SMC
  plot-gui           : Mở giao diện thanh trượt tinh chỉnh Live Tuning

⚙️ [BIÊN DỊCH & CẬP NHẬT]
  reload (capnhat)   : Nạp lại toàn bộ lệnh mới nhất sau khi git pull
  git-sync (dongbo)  : Kéo code mới nhất từ GitHub + Build lại tự động 1-Click
  build              : Build nhanh workspace (colcon build)
  build-all          : Build toàn bộ tất cả package
  ros-help           : Xem lại bảng hướng dẫn này bất cứ lúc nào

📡 [CHUYỂN MẠNG & TÌM PI] (Hotspot điện thoại / Mạng trường)
  edit-net (sua-mang): Mở file cấu hình mạng network_config.env để điền tên/pass Wi-Fi
  find-pi (tim-pi)   : Tự động tìm IP của Pi trên bất kỳ mạng nào (Chạy trên PC)
  switch-hotspot     : Chuyển Pi sang Hotspot điện thoại (Chạy trên Pi SSH)
  switch-school      : Chuyển Pi về mạng trường CTU (Chạy trên Pi SSH)
  check-wifi (xem-mang): Xem mạng Wi-Fi đang kết nối và IP hiện tại
================================================================================
EOF
}
unalias ros-help 2>/dev/null || true
unalias help-robot 2>/dev/null || true
unalias robot-help 2>/dev/null || true
alias ros-help="ros_help_func"
alias robot-help="ros_help_func"
alias help-robot="ros_help_func"

# Lệnh DỪNG KHẨN CẤP & DỌN SẠCH TIẾN TRÌNH ROBOT (Dùng khi Ctrl+C hoặc xe bị nhiễu)
stop_robot_func() {
    echo "🛑 Đang gửi lệnh phanh khẩn cấp & dừng toàn bộ động cơ..."
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" 2>/dev/null || true
    echo "🧹 Đang dọn sạch các tiến trình ROS 2 còn sót lại..."
    killall -9 rplidar_node sllidar_node esp32_bridge imu_driver camera_publisher wifi_cam_bridge cnn_driver costmap_node async_slam_toolbox_node 2>/dev/null || true
    fuser -k /dev/video* 2>/dev/null || true
    echo "✅ Toàn bộ hệ thống Robot đã dừng an toàn và giải phóng cổng Serial / Camera!"
}
alias stop-robot="stop_robot_func"
alias stop="stop_robot_func"
alias dung="stop_robot_func"

# Lệnh HỦY DẪN ĐƯỜNG NAV2 (Hủy mục tiêu điểm đến phím G & phanh dừng xe)
cancel_nav_func() {
    echo "🛑 Đang hủy mục tiêu dẫn đường Nav2 & dừng xe..."
    ros2 service call /navigate_to_pose/_action/cancel_goal action_msgs/srv/CancelGoal "{goal_info: {goal_id: {uuid: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}, stamp: {sec: 0, nanosec: 0}}}" 2>/dev/null || true
    ros2 service call /navigate_through_poses/_action/cancel_goal action_msgs/srv/CancelGoal "{goal_info: {goal_id: {uuid: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}, stamp: {sec: 0, nanosec: 0}}}" 2>/dev/null || true
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" 2>/dev/null || true
    echo "✅ Đã hủy lệnh Nav2! Xe đã dừng an toàn."
}
alias huy="cancel_nav_func"
alias huy-nav="cancel_nav_func"
alias stop-nav="cancel_nav_func"
alias cancel-nav="cancel_nav_func"

# Tự động nạp môi trường workspace ngay khi mở terminal
load_ws

