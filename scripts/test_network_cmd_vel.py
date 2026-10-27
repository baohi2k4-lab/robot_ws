#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelTester(Node):
    def __init__(self, mode='pub'):
        super().__init__('cmd_vel_tester')
        self.mode = mode
        if mode == 'pub':
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.timer = self.create_timer(0.2, self.send_cmd)
            self.count = 0
            self.get_logger().info("Tester Publisher started on /cmd_vel")
        else:
            self.sub = self.create_subscription(Twist, '/cmd_vel', self.recv_cmd, 10)
            self.received_count = 0
            self.get_logger().info("Tester Subscriber started on /cmd_vel")

    def send_cmd(self):
        msg = Twist()
        msg.linear.x = 0.25
        msg.angular.z = -0.15
        self.pub.publish(msg)
        self.count += 1
        self.get_logger().info(f"Published /cmd_vel #{self.count}: v={msg.linear.x}, w={msg.angular.z}")
        if self.count >= 15:
            rclpy.shutdown()

    def recv_cmd(self, msg: Twist):
        self.received_count += 1
        self.get_logger().info(f"RECEIVED /cmd_vel #{self.received_count}: v={msg.linear.x}, w={msg.angular.z}")
        if self.received_count >= 3:
            print(f"SUCCESS: Received {self.received_count} packets on /cmd_vel!")
            rclpy.shutdown()

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'pub'
    rclpy.init()
    node = CmdVelTester(mode)
    rclpy.spin(node)
