#!/usr/bin/env python3
"""
==============================================================================
  Launch file: laptop_ai.launch.py (Package: my_robot_bringup)
  
  Mục đích:
    Khởi chạy toàn bộ Camera cắm trên Laptop + Hệ thống AI CNN điều khiển xe thật.
    - Camera (Intel RealSense D435 / iPhone) cắm trực tiếp vào cổng USB của Laptop.
    - Model ONNX chạy trực tiếp trên GPU/CPU Laptop đạt 30 - 60 FPS (độ trễ ~10ms).
    - Nhận /odometry/filtered, /scan, /gps/vel từ Raspberry Pi qua mạng (< 100 KB/s).
    - Tính toán góc lái (SMC / Pure Pursuit) và phát lệnh /cmd_vel sang Pi để lái 4 bánh xe.
    
  Sử dụng trên Laptop:
    ros2 launch my_robot_bringup laptop_ai.launch.py
    (hoặc dùng lệnh tắt: laptop-ai)
==============================================================================
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('my_robot_bringup')
    pkg_ctrl = get_package_share_directory('my_robot_controller')

    params_file = os.path.join(pkg_ctrl, 'config', 'params_real.yaml')

    # ── Arguments ──────────────────────────────────────────────────────────
    camera_device_arg = DeclareLaunchArgument(
        'camera_device', default_value='auto',
        description='Cổng camera USB trên Laptop (/dev/video0, auto, URL iPhone)'
    )
    width_arg = DeclareLaunchArgument(
        'width', default_value='640',
        description='Chiều rộng khung hình camera (640 hoặc 1280)'
    )
    height_arg = DeclareLaunchArgument(
        'height', default_value='480',
        description='Chiều cao khung hình camera (480 hoặc 720)'
    )
    fps_arg = DeclareLaunchArgument(
        'fps', default_value='30.0',
        description='Tốc độ khung hình camera FPS'
    )
    enable_gps_arg = DeclareLaunchArgument(
        'enable_gps', default_value='true',
        description='Kích hoạt đo trượt Doppler và đánh giá quay đầu Omega Turn bằng GPS'
    )

    # ── 1. Camera Publisher (Chạy trên Laptop, đọc USB RealSense D435 / iPhone) ──
    camera_publisher_node = Node(
        package='my_robot_bringup',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'camera_frame_id': 'camera_link',
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'fps': LaunchConfiguration('fps'),
        }]
    )

    # ── 2. AI CNN Driver (Chạy ONNX trên Laptop 30-60 FPS, điều khiển xe Pi) ──
    cnn_driver_node = Node(
        package='my_robot_controller',
        executable='cnn_driver',
        name='cnn_driver_node',
        output='screen',
        parameters=[
            params_file,
            {
                'image_topic': '/camera/color/image_raw',
                'odom_topic': '/odometry/filtered',
                'scan_topic': '/scan',
                'cmd_vel_topic': '/cmd_vel',
                'gps_vel_topic': '/gps/vel',
                'enable_gps': LaunchConfiguration('enable_gps'),
                'use_sim_time': False,
            }
        ]
    )

    return LaunchDescription([
        camera_device_arg,
        width_arg,
        height_arg,
        fps_arg,
        enable_gps_arg,
        camera_publisher_node,
        cnn_driver_node,
    ])
