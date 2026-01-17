---
# Translation needed
# اس صفحے کا ترجمہ درکار ہے
---

**This page is currently only available in English. Please contribute a translation.**
**یہ صفحہ فی الحال صرف انگریزی میں دستیاب ہے۔ براہ کرم ایک ترجمہ شامل کریں۔**

---
id: module-1-topics-message-passing
title: Chapter 3 - Topics and Message Passing
sidebar_label: Chapter 3 - Topics and Message Passing
---

# Chapter 3: Topics and Message Passing

## Publisher-Subscriber Pattern

The publisher-subscriber pattern is one of the fundamental communication mechanisms in ROS2. It enables asynchronous, one-to-many communication where publishers send messages to topics without knowing which subscribers will receive them. This decoupling allows for flexible system architectures where components can be added or removed without affecting others.

### Core Concepts

In the publisher-subscriber model:
- **Publishers** send messages to named topics
- **Subscribers** receive messages from named topics
- **Topics** serve as communication channels that connect publishers and subscribers
- Communication is asynchronous and non-blocking

This pattern is ideal for streaming data like sensor readings, robot state information, or other continuous data streams where the sender doesn't need to know who receives the data.

## Message Types and Definitions

### Standard Message Types

ROS2 provides a rich set of standard message types organized in several packages:

- **std_msgs**: Basic data types like String, Int32, Float64, etc.
- **geometry_msgs**: Geometric primitives like Point, Pose, Vector3, etc.
- **sensor_msgs**: Sensor-specific messages like LaserScan, Image, JointState, etc.
- **nav_msgs**: Navigation-related messages like Odometry, Path, etc.
- **action_msgs**: Action-specific messages for goal and feedback

### Creating Custom Message Types

Custom message types allow you to define application-specific data structures. Messages are defined using the `.msg` file format with a simple syntax:

```
# CustomHumanoidState.msg
string robot_name
float64[20] joint_positions
float64[20] joint_velocities
geometry_msgs/Pose current_pose
sensor_msgs/BatteryState battery_status
```

To create a custom message:
1. Create a `msg` directory in your package
2. Define your message in a `.msg` file
3. Update your `CMakeLists.txt` and `package.xml` to include the message
4. Build your package to generate the message headers/classes

### Message Definition Syntax

Message definitions follow these rules:
- Each line defines a field with type and name
- Built-in types: bool, byte, char, float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, string, wstring
- Arrays are defined with fixed size `[n]` or unbounded `[]`
- Constants can be defined at the top with `TYPE CONSTANT_NAME=VALUE`

## Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior to match application requirements. ROS2 provides several QoS policies:

### Reliability Policy
- **RELIABLE**: Ensures all messages are delivered (uses acknowledgments)
- **BEST_EFFORT**: Attempts delivery without guarantees (faster but may lose messages)

### Durability Policy
- **TRANSIENT_LOCAL**: Publisher keeps historical data for late-joining subscribers
- **VOLATILE**: No historical data is maintained

### History Policy
- **KEEP_LAST**: Maintain a fixed number of most recent messages
- **KEEP_ALL**: Maintain all messages (limited by available memory)

### Example QoS Configuration

```cpp
// C++ example with custom QoS
rclcpp::QoS qos_profile(10);  // history depth of 10
qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

auto publisher = this->create_publisher<std_msgs::msg::String>(
    "topic_name", qos_profile);
```

```python
# Python example with custom QoS
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)

publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

## Real-time Communication Patterns

### Synchronous vs Asynchronous Communication

Topics provide asynchronous communication by default, which is beneficial for real-time systems as it prevents blocking operations. However, you can implement synchronization patterns when needed:

```cpp
// Example of synchronized message handling
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// Synchronize messages from multiple sensors
message_filters::Subscriber<sensor_msgs::msg::LaserScan> laser_sub(this, "laser_scan");
message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_sub(this, "pointcloud");

typedef message_filters::TimeSynchronizer<sensor_msgs::msg::LaserScan, sensor_msgs::msg::PointCloud2> Sync;
auto sync = std::make_shared<Sync>(laser_sub, pointcloud_sub, 10);
sync->registerCallback(&MyNode::syncCallback, this);
```

### Time-Critical Applications

For time-critical applications, consider these patterns:

- Use appropriate QoS settings for your timing requirements
- Implement message timestamps for temporal analysis
- Use keep-all history policy for critical data
- Consider using reliable reliability policy for critical messages

## Custom Message Example for Humanoid Robot

Let's create a custom message for humanoid robot state:

```
# HumanoidState.msg
string robot_name
# Joint information
float64[] joint_positions
float64[] joint_velocities
float64[] joint_efforts
string[] joint_names
# Balance information
geometry_msgs/Vector3 center_of_mass
geometry_msgs/Vector3 zero_moment_point
bool is_balanced
# Task information
string current_task
float64 task_progress
```

### Publisher Example with Custom Message

#### C++ Publisher Example

```cpp
// C++ publisher with custom message and QoS
#include "my_robot_msgs/msg/humanoid_state.hpp"

class HumanoidStatePublisher : public rclcpp::Node
{
public:
    HumanoidStatePublisher()
    : Node("humanoid_state_publisher")
    {
        // Create publisher with specific QoS for real-time performance
        rclcpp::QoS qos_profile(5);
        qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        qos_profile.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

        publisher_ = this->create_publisher<my_robot_msgs::msg::HumanoidState>(
            "humanoid_state", qos_profile);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 10 Hz update rate
            std::bind(&HumanoidStatePublisher::publish_state, this));
    }

private:
    void publish_state()
    {
        auto msg = my_robot_msgs::msg::HumanoidState();
        msg.robot_name = "my_humanoid_robot";
        msg.joint_names = {"hip_left", "knee_left", "ankle_left", /* ... */};
        msg.joint_positions = {0.1, 0.2, 0.3, /* ... */};
        msg.joint_velocities = {0.0, 0.0, 0.0, /* ... */};
        msg.joint_efforts = {0.0, 0.0, 0.0, /* ... */};
        msg.is_balanced = true;
        msg.current_task = "walking";
        msg.task_progress = 0.5;

        publisher_->publish(msg);
    }

    rclcpp::Publisher<my_robot_msgs::msg::HumanoidState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

#### Python Publisher Example

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from my_robot_msgs.msg import HumanoidState  # Assuming custom message is generated


class HumanoidStatePublisher(Node):

    def __init__(self):
        super().__init__('humanoid_state_publisher')

        # Create QoS profile for real-time performance
        qos_profile = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(
            HumanoidState,
            'humanoid_state',
            qos_profile
        )

        # Create timer for 10 Hz update rate
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_state)

        self.get_logger().info('Humanoid State Publisher initialized')

    def publish_state(self):
        msg = HumanoidState()
        msg.robot_name = 'my_humanoid_robot'
        msg.joint_names = ['hip_left', 'knee_left', 'ankle_left']  # Simplified for example
        msg.joint_positions = [0.1, 0.2, 0.3]
        msg.joint_velocities = [0.0, 0.0, 0.0]
        msg.joint_efforts = [0.0, 0.0, 0.0]
        msg.is_balanced = True
        msg.current_task = 'walking'
        msg.task_progress = 0.5

        self.publisher.publish(msg)
        self.get_logger().info(f'Published state for robot: {msg.robot_name}')


def main(args=None):
    rclpy.init(args=args)
    publisher = HumanoidStatePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Subscriber Example with Custom Message

#### C++ Subscriber Example

```cpp
// C++ subscriber with custom message
class HumanoidStateSubscriber : public rclcpp::Node
{
public:
    HumanoidStateSubscriber()
    : Node("humanoid_state_subscriber")
    {
        // Use matching QoS profile
        rclcpp::QoS qos_profile(5);
        qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        qos_profile.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

        subscription_ = this->create_subscription<my_robot_msgs::msg::HumanoidState>(
            "humanoid_state", qos_profile,
            std::bind(&HumanoidStateSubscriber::state_callback, this, std::placeholders::_1));
    }

private:
    void state_callback(const my_robot_msgs::msg::HumanoidState::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(),
            "Received state for robot: %s, task: %s, progress: %.2f",
            msg->robot_name.c_str(),
            msg->current_task.c_str(),
            msg->task_progress);
    }

    rclcpp::Subscription<my_robot_msgs::msg::HumanoidState>::SharedPtr subscription_;
};
```

#### Python Subscriber Example

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from my_robot_msgs.msg import HumanoidState  # Assuming custom message is generated


class HumanoidStateSubscriber(Node):

    def __init__(self):
        super().__init__('humanoid_state_subscriber')

        # Create matching QoS profile
        qos_profile = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            HumanoidState,
            'humanoid_state',
            self.state_callback,
            qos_profile
        )

        self.subscription  # prevent unused variable warning
        self.get_logger().info('Humanoid State Subscriber initialized')

    def state_callback(self, msg):
        self.get_logger().info(
            f'Received state for robot: {msg.robot_name}, '
            f'task: {msg.current_task}, progress: {msg.task_progress:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    subscriber = HumanoidStateSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Topic Communication

### Performance Considerations
- Use appropriate message sizes to avoid network congestion
- Select QoS policies based on your application's requirements
- Consider message compression for large data like images
- Use appropriate update rates for your application

### Design Patterns
- Separate high-frequency and low-frequency topics
- Use dedicated topics for time-critical data
- Implement data validation in subscribers
- Consider using latching for static data

## Diagram: Topic Communication Flow

```
                    ROS2 Communication System
                    +-------------------------+
                    |        DDS Layer        |
                    +-------------------------+
                             |    |
                    Publisher|    |Subscriber
                    +--------+    +--------+
                    |  Node A         Node B |
                    |                       |
                    |  +--------------+     |
                    |  |  Publisher   |     |
                    |  |              |     |
                    |  |  Topic:      |     |
                    |  |  /joint_states|     |
                    |  +--------------+     |
                    |                       |
                    |  +--------------+     |
                    |  |  Subscriber  |<----+----+
                    |  |              |     |    |
                    |  |  Topic:      |     |    |
                    |  |  /joint_states|     |    |
                    |  +--------------+     |    |
                    +-----------------------+    |
                                                 |
                    +-----------------------+    |
                    |  Node C               |    |
                    |                       |    |
                    |  +--------------+     |    |
                    |  |  Publisher   |-----+----+
                    |  |              |
                    |  |  Topic:      |
                    |  |  /joint_states|
                    |  +--------------+
                    +-----------------------+
```

## Testing Examples

All the code examples provided in this chapter have been designed to work with the ROS2 Humble Hawksbill distribution. To test these examples:

1. Create a new ROS2 package: `ros2 pkg create --build-type ament_cmake my_robot_msgs`
2. Add your custom message definition to the `msg` directory
3. Update your `CMakeLists.txt` and `package.xml` to include message generation
4. Build your workspace: `colcon build --packages-select my_robot_msgs`
5. Source your workspace: `source install/setup.bash`
6. Run your publisher and subscriber nodes to verify communication

## Conclusion

Topics provide the backbone of ROS2 communication, enabling flexible, decoupled system architectures. Understanding QoS settings is crucial for achieving the right balance between reliability, performance, and resource usage. Custom messages allow you to tailor communication to your specific application needs, while proper design patterns ensure efficient and robust communication in your robotic systems.

The publisher-subscriber pattern is particularly well-suited for humanoid robotics applications where sensor data, joint states, and other continuous information need to be shared among multiple components without tight coupling.