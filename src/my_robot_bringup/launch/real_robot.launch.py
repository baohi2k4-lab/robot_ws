"""
Launch file khởi động TOÀN BỘ HỆ THỐNG XE THẬT (Raspberry Pi / Laptop).
Bao gồm:
  - Robot State Publisher (URDF 3D model + TF tree)
  - Joint State Publisher
  - RPLIDAR C1 (sllidar_ros2, laser_frame, 460800 baud)
  - Static Odom TF (hoặc ESP32 Hardware Bridge nếu enable_esp32:=true)
  - USB Webcam Camera (tùy chọn enable_camera:=true qua v4l2_camera)
  - CNN Driver (tùy chọn enable_cnn:=true)
  - RViz2 (tùy chọn enable_rviz:=true)

Sử dụng:
  ros2 launch my_robot_bringup real_robot.launch.py
  ros2 launch my_robot_bringup real_robot.launch.py serial_port:=/dev/ttyUSB0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('my_robot_bringup')
    pkg_desc = get_package_share_directory('my_robot_description')
    pkg_ctrl = get_package_share_directory('my_robot_controller')

    xacro_file = os.path.join(pkg_desc, 'urdf', 'robot.urdf.xacro')
    rviz_config = os.path.join(pkg_bringup, 'rviz', 'real_robot.rviz')
    params_real = os.path.join(pkg_ctrl, 'config', 'params_real.yaml')
    ekf_config = os.path.join(pkg_bringup, 'config', 'ekf.yaml')

    robot_description = Command(['xacro ', '"', xacro_file, '"'])

    # ── Launch Arguments ─────────────────────────────────────────────
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/rplidar',
        description='RPLIDAR C1 serial port')

    camera_device_arg = DeclareLaunchArgument(
        'camera_device', default_value='/dev/video0',
        description='USB camera device path (/dev/video0)')

    esp32_port_arg = DeclareLaunchArgument(
        'esp32_port', default_value='/dev/esp32',
        description='ESP32 serial port')

    enable_esp32_arg = DeclareLaunchArgument(
        'enable_esp32', default_value='true',
        description='Enable ESP32 motor/odom bridge (if false, publishes static odom TF)')

    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera', default_value='true',
        description='Enable USB camera node')

    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='true',
        description='Enable RPLIDAR C1 sensor node')



    enable_cnn_arg = DeclareLaunchArgument(
        'enable_cnn', default_value='false',
        description='Enable CNN row-following driver')

    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz', default_value='false',
        description='Enable RViz2 visualization')

    record_arg = DeclareLaunchArgument(
        'record', default_value='false',
        description='Record raw camera video to MP4 dataset file on Pi')

    record_name_arg = DeclareLaunchArgument(
        'record_name', default_value='',
        description='Custom filename for recorded MP4 video')

    enable_imu_arg = DeclareLaunchArgument(
        'enable_imu', default_value='true',
        description='Enable ICM-20948 9-axis IMU driver over I2C')

    enable_madgwick_arg = DeclareLaunchArgument(
        'enable_madgwick', default_value='true',
        description='Enable Madgwick orientation filter for IMU')

    enable_ekf_arg = DeclareLaunchArgument(
        'enable_ekf', default_value='true',
        description='Enable EKF sensor fusion (Wheel Odometry + IMU)')

    enable_gps_arg = DeclareLaunchArgument(
        'enable_gps', default_value='true',
        description='Enable GPS NEO-M10, NavSat Transform and EKF 2 Global')

    gps_port_arg = DeclareLaunchArgument(
        'gps_port', default_value='/dev/ttyAMA0',
        description='Serial port for GPS NEO-M10 (default: /dev/ttyAMA0)')

    gps_baud_arg = DeclareLaunchArgument(
        'gps_baud', default_value='38400',
        description='Baudrate for GPS NEO-M10 (default: 38400)')

    enable_costmap_arg = DeclareLaunchArgument(
        'enable_costmap', default_value='true',
        description='Enable real-time costmap generation for RViz visualization')

    # ── Robot State Publisher (URDF + TF) ────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='log',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}]
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='log',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
            'rate': 30
        }]
    )

    # ── Static TF: odom -> base_footprint (Khi không có encoder ESP32 và không có EKF) ─
    static_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_odom_base',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'odom', '--child-frame-id', 'base_footprint'],
        condition=UnlessCondition(
            PythonExpression(["'", LaunchConfiguration('enable_esp32'), "' == 'true' or '", LaunchConfiguration('enable_ekf'), "' == 'true'"])
        )
    )

    # ── RPLIDAR C1 (Official sllidar_ros2) ───────────────────────────
    lidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        output='screen',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baudrate': 460800,
            'frame_id': 'laser_frame',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Standard',
        }],
        condition=IfCondition(LaunchConfiguration('enable_lidar'))
    )

    # ── Camera Driver (OpenCV MJPG — thay thế v4l2_camera vì nó không giải mã được MJPG) ─
    camera_publisher_node = Node(
        package='my_robot_bringup',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'width': 640,
            'height': 480,
            'fps': 15.0,
            'camera_frame_id': 'camera_link',
        }],
        condition=IfCondition(LaunchConfiguration('enable_camera'))
    )

    # ── WiFi Camera Bridge (Nén JPEG gửi qua Wi-Fi) ─────────────────
    wifi_cam_bridge = Node(
        package='my_robot_bringup',
        executable='wifi_cam_bridge',
        name='wifi_cam_bridge',
        output='screen',
        parameters=[{
            'target_width': 320,
            'target_height': 240,
            'jpeg_quality': 50,
            'skip_frames': 2,
        }],
        condition=IfCondition(LaunchConfiguration('enable_camera'))
    )

    # ── ESP32 Hardware Bridge (Motor PID + Encoder Odom) ─────────────
    esp32_bridge = Node(
        package='my_robot_bringup',
        executable='esp32_bridge',
        name='esp32_bridge',
        output='screen',
        parameters=[{
            'connection_mode': 'serial',
            'serial_port': LaunchConfiguration('esp32_port'),
            'baudrate': 115200,
            'wheel_diameter': 0.20,
            'wheel_base': 0.58,
            'encoder_ppr': 200,
            'quadrature': 4,
            'gear_ratio': 1.0,
            'publish_tf': PythonExpression(["'false' if '", LaunchConfiguration('enable_ekf'), "' == 'true' else 'true'"]),
            'odom_topic': PythonExpression(["'/wheel/odom' if '", LaunchConfiguration('enable_ekf'), "' == 'true' else '/odom'"]),
            'encoder_sign': -1.0,
        }],
        condition=IfCondition(LaunchConfiguration('enable_esp32'))
    )

    # ── CNN Driver (Autonomous Row Following) ────────────────────────
    cnn_driver = Node(
        package='my_robot_controller',
        executable='cnn_driver',
        name='cnn_driver_node',
        output='screen',
        parameters=[
            params_real,
            {
                'image_topic': '/camera/color/image_raw',
                'odom_topic': PythonExpression(["'/odometry/filtered' if '", LaunchConfiguration('enable_ekf'), "' == 'true' else '/odom'"]),
                'imu_topic': '/imu/data',
            }
        ],
        remappings=[
            ('camera/image_raw', '/camera/color/image_raw'),
            ('odom', PythonExpression(["'/odometry/filtered' if '", LaunchConfiguration('enable_ekf'), "' == 'true' else '/odom'"])),
            ('/imu', '/imu/data'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_cnn'))
    )

    # ── Raw Video Recorder (Chỉ bật khi record:=true để ghi dataset thô) ─
    video_recorder = Node(
        package='my_robot_bringup',
        executable='video_recorder',
        name='video_recorder',
        output='screen',
        parameters=[{
            'device': LaunchConfiguration('camera_device'),
            'topic': '/camera/color/image_raw',
            'filename': LaunchConfiguration('record_name'),
            'width': 640,
            'height': 480,
            'fps': 30.0,
            'direct_capture': True,
        }],
        condition=IfCondition(LaunchConfiguration('record'))
    )

    # ── ICM-20948 IMU Node (I2C trên Raspberry Pi) ───────────────────
    imu_node = Node(
        package='my_robot_controller',
        executable='imu_driver',
        name='imu_driver_node',
        output='screen',
        parameters=[{
            'i2c_bus': 1,
            'i2c_address': 0x68,
            'frame_id': 'imu_link',
            'publish_topic': '/imu/data',
            'raw_topic': '/imu/data_raw',
            'rate_hz': 50.0,
            # Cấu hình lọc rung động cơ 775 + hộp số và bù trôi Zero-bias (ZUPT)
            'vibration_filter_enabled': True,
            'accel_ema_alpha': 0.75,
            'gyro_ema_alpha': 0.80,
            'adaptive_bias_tracking': True,
            'stationary_speed_threshold': 0.02,
            'mount_upside_down': True,
        }],
        condition=IfCondition(LaunchConfiguration('enable_imu'))
    )

    # ── Madgwick AHRS Orientation Filter (imu_tools) ─────────────────
    madgwick_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick_node',
        output='log',
        parameters=[{
            'use_mag': False,
            'publish_tf': False,
            'world_frame': 'enu',
            'gain': 0.1,
            'zeta': 0.0,
        }],
        remappings=[
            ('imu/data_raw', '/imu/data_raw'),
            ('imu/data', '/imu/data'),
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('enable_imu'), "' == 'true' and '", LaunchConfiguration('enable_madgwick'), "' == 'true'"])
        )
    )

    # ── DUAL EKF + NAVSAT TRANSFORM (REP-105: EKF 1 Local + NavSat + EKF 2 Global) ─
    dual_ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'dual_ekf.launch.py')
        ),
        launch_arguments={
            'enable_gps': LaunchConfiguration('enable_gps'),
            'gps_port': LaunchConfiguration('gps_port'),
            'gps_baud': LaunchConfiguration('gps_baud'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_ekf'))
    )

    # ── Real-time Costmap Node (Vẽ bản đồ chi phí thời gian thực cho RViz) ─
    costmap_node = Node(
        package='my_robot_navigation',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[
            params_real,
            {
                'inscribed_radius': 0.35,
                'inflation_radius': 0.45,
                'cost_scaling_factor': 10.0,
                'obstacle_threshold': 40,
                'resolution': 0.05,
                'map_length_m': 60.0,
                'map_width_m': 20.0,
                'publish_rate': 4.0,
                'global_frame': 'odom',
                'rolling_window': False,
                'use_sim_time': False,
            }
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('enable_costmap'), "' == 'true' and '", LaunchConfiguration('enable_lidar'), "' == 'true'"])
        )
    )

    # ── RViz2 ────────────────────────────────────────────────────────
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('enable_rviz'))
    )

    return LaunchDescription([
        serial_port_arg,
        camera_device_arg,
        esp32_port_arg,
        enable_esp32_arg,
        enable_camera_arg,
        enable_lidar_arg,
        enable_costmap_arg,
        enable_cnn_arg,
        enable_rviz_arg,
        record_arg,
        record_name_arg,
        enable_imu_arg,
        enable_madgwick_arg,
        enable_ekf_arg,
        enable_gps_arg,
        gps_port_arg,
        gps_baud_arg,
        robot_state_pub,
        joint_state_pub,
        static_odom_tf,
        lidar_node,
        camera_publisher_node,
        wifi_cam_bridge,
        esp32_bridge,
        imu_node,
        madgwick_node,
        dual_ekf_launch,
        video_recorder,
        cnn_driver,
        costmap_node,
        rviz2_node,
    ])
