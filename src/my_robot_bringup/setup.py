from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_robot_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vinh',
    maintainer_email='vinh@todo.todo',
    description='Bringup package for real robot hardware',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'esp32_bridge = my_robot_bringup.esp32_bridge:main',
            'wifi_cam_bridge = my_robot_bringup.wifi_cam_bridge:main',
            'wifi_cam_receiver = my_robot_bringup.wifi_cam_receiver:main',
            'video_recorder = my_robot_bringup.video_recorder:main',
            'camera_recorder = my_robot_bringup.camera_recorder:main',
            'camera_publisher = my_robot_bringup.camera_publisher:main',
            'laser_filter = my_robot_bringup.laser_filter:main',
        ],
    },
)
