#!/usr/bin/env python3
"""
==============================================================================
  Launch File: dual_ekf.launch.py (Package: my_robot_bringup)
  
  Khởi chạy kiến trúc 2 tầng EKF (REP-105):
  1. EKF 1 (LOCAL)  : Dung hợp /wheel/odom + /imu/data -> Xuất /odometry/local
                      Broadcast TF: odom -> base_footprint (Mượt, không giật bước)
  2. NavSat Transform: Chuyển đổi /gps/fix + /imu/data + /odometry/local -> /odometry/gps
  3. EKF 2 (GLOBAL) : Dung hợp /wheel/odom + /imu/data + /odometry/gps -> Xuất /odometry/global
                      Broadcast TF: map -> odom (Triệt tiêu trôi tích lũy bằng GPS)
==============================================================================
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('my_robot_bringup')
    dual_ekf_config = os.path.join(pkg_bringup, 'config', 'dual_ekf_navsat.yaml')

    # Arguments
    enable_gps_arg = DeclareLaunchArgument(
        'enable_gps', default_value='false',
        description='Bật EKF 2 (Global) và NavSat Transform dung hợp GPS'
    )
    gps_port_arg = DeclareLaunchArgument(
        'gps_port', default_value='/dev/ttyAMA0',
        description='Cổng Serial module GPS NEO-M10 (mặc định /dev/ttyAMA0)'
    )
    gps_baud_arg = DeclareLaunchArgument(
        'gps_baud', default_value='38400',
        description='Baudrate GPS NEO-M10 (mặc định 38400)'
    )

    # ── 1. EKF 1 (LOCAL): Dung hợp Wheel Odom + IMU -> /odometry/local ────────
    ekf_local_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_local',
        output='screen',
        parameters=[dual_ekf_config],
        remappings=[
            ('odometry/filtered', '/odometry/filtered'),
            ('accel/filtered', '/accel/filtered'),
        ]
    )

    # ── 2. GPS Driver: Đọc NMEA từ NEO-M10 -> /gps/fix ────────────────────────
    gps_driver_node = Node(
        package='my_robot_controller',
        executable='gps_driver',
        name='gps_driver_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('gps_port'),
            'baudrate': LaunchConfiguration('gps_baud'),
            'frame_id': 'gps_link',
            'publish_topic': '/gps/fix',
        }],
        condition=IfCondition(LaunchConfiguration('enable_gps'))
    )

    # ── 3. NavSat Transform: GPS Fix -> Tọa độ Descartes /odometry/gps ─────────
    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[dual_ekf_config],
        remappings=[
            ('imu', '/imu/data'),
            ('gps/fix', '/gps/fix'),
            ('odometry/filtered', '/odometry/filtered'),
            ('odometry/gps', '/odometry/gps'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_gps'))
    )

    # ── 4. EKF 2 (GLOBAL): Dung hợp Thêm GPS -> /odometry/global ──────────────
    ekf_global_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_global',
        output='screen',
        parameters=[dual_ekf_config],
        remappings=[
            ('odometry/filtered', '/odometry/global'),
            ('accel/filtered', '/accel/global'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_gps'))
    )

    return LaunchDescription([
        enable_gps_arg,
        gps_port_arg,
        gps_baud_arg,
        ekf_local_node,
        gps_driver_node,
        navsat_transform_node,
        ekf_global_node,
    ])

