import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from rclpy.timer import Timer
from std_msgs.msg import String
from lifecycle_msgs.msg import State


class LifecycleNodeExample(LifecycleNode):

    def __init__(self):
        super().__init__('lifecycle_node')
        self.get_logger().info('Lifecycle node created')
        self.pub = None
        self.timer = None

    def on_configure(self, state):
        self.get_logger().info('Configuring node')
        # Initialize resources but don't start processing
        self.pub = self.create_publisher(String, 'lifecycle_topic', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node')
        # Start processing, activate publishers/subscribers
        self.pub.on_activate()

        # Create a timer to publish messages while active
        self.timer = self.create_timer(1.0, self.timer_callback)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node')
        # Stop processing, deactivate publishers/subscribers
        self.pub.on_deactivate()
        if self.timer:
            self.timer.cancel()
            self.timer = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up node')
        # Release resources
        if self.pub:
            self.destroy_publisher(self.pub)
            self.pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info('Shutting down node')
        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        if self.get_current_state().id() == State.PRIMARY_STATE_ACTIVE:
            msg = String()
            msg.data = 'Lifecycle message'
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LifecycleNodeExample()

    # Manually trigger lifecycle transitions for demonstration
    node.configure()
    node.activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()