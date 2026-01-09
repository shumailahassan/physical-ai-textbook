import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class CustomMessagePublisher(Node):

    def __init__(self):
        super().__init__('custom_message_publisher')

        # Create publisher with custom QoS settings
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(JointState, 'humanoid_joints', qos_profile)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # 10 Hz

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['hip_left', 'knee_left', 'ankle_left', 'hip_right', 'knee_right', 'ankle_right']
        msg.position = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CustomMessagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()