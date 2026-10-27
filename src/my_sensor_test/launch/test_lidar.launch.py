#!/usr/bin/env python3
"""
Launch file to test real LiDAR RPLIDAR C1 (C1M1) with 3D Robot Model in RViz2.
Displays:
1. 3D Robot Model (with wheels, chassis, axes at origin)
2. RPLIDAR C1 red laser scan points (matching robot_ws simulation aesthetics)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sensor_test = get_package_share_directory('my_sensor_test')
    pkg_desc = get_package_share_directory('my_robot_description')

    rviz_config = os.path.join(pkg_sensor_test, 'rviz', 'test_sensors.rviz')
    xacro_file = os.path.join(pkg_desc, 'urdf', 'robot.urdf.xacro')

    robot_description = Command(['xacro ', '"', xacro_file, '"'])

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/rplidar',
        description='LiDAR USB serial port (/dev/rplidar or /dev/ttyUSB0)'
    )

    serial_baudrate_arg = DeclareLaunchArgument(
        'serial_baudrate',
        default_value='460800',
        description='LiDAR baudrate (C1: 460800, A1/A2: 115200)'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to launch RViz2 directly on this machine'
    )

    # Robot State Publisher (publishes 3D robot model & TF)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}]
    )

    # Joint State Publisher (publishes wheel joints)
    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    # Official RPLIDAR C1 Node
    lidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        output='screen',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baudrate': LaunchConfiguration('serial_baudrate'),
            'frame_id': 'laser_frame',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Standard',
        }],
        remappings=[
            ('scan', '/scan_raw'),
        ]
    )

    # Laser Filter Node (Filters pillar & chassis blind spot)
    laser_filter_node = Node(
        package='my_robot_bringup',
        executable='laser_filter',
        name='laser_filter_node',
        output='screen',
        parameters=[{
            'input_topic': '/scan_raw',
            'output_topic': '/scan',
            'min_distance': 0.18,
            'max_distance': 12.0,
            'filter_rear_cone': True,
            'rear_angle_deg': 30.0,
            'rear_dist_limit': 0.35,
        }]
    )

    # Sensor Visualizer Node (Range rings, directions, obstacle badges)
    visualizer_node = Node(
        package='my_sensor_test',
        executable='sensor_visualizer',
        name='sensor_visualizer',
        output='screen'
    )

    # RViz2 Node (Only launched if use_rviz:=true)
    from launch.conditions import IfCondition
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        serial_port_arg,
        serial_baudrate_arg,
        use_rviz_arg,
        robot_state_pub,
        joint_state_pub,
        lidar_node,
        laser_filter_node,
        visualizer_node,
        rviz2_node,
    ])

