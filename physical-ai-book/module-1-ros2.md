# Module 1 — The Robotic Nervous System (ROS 2)

## Introduction

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's not an operating system but rather a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. Think of ROS 2 as the "nervous system" of a robot, coordinating communication between different components and enabling them to work together seamlessly.

ROS 2 provides distributed computing capabilities, allowing different parts of a robot's software to run on different computers while maintaining reliable communication. This is essential for humanoid robots, which often have multiple sensors, actuators, and processing units that need to coordinate their actions in real-time.

Key features of ROS 2 include:
- A rich ecosystem of packages and tools
- Support for multiple programming languages (C++, Python, etc.)
- Real-time performance capabilities
- Improved security and safety features
- Better support for multi-robot systems
- Standardized message formats for interoperability

## ROS 2 Nodes

A ROS 2 node is a fundamental building block of a ROS 2 program. It's essentially a process that performs computation and communicates with other nodes through topics, services, and actions. Each node can publish to topics, subscribe to topics, offer services, or call services.

### Understanding Nodes

Nodes are organized in a distributed fashion, meaning they can run on different machines but still communicate with each other. This architecture allows for:

- **Modularity**: Each node can focus on a specific task
- **Scalability**: Additional nodes can be added without disrupting existing functionality
- **Fault tolerance**: If one node fails, others can continue operating

### Creating a Simple Node in Python

Here's an example of a basic ROS 2 node written in Python:

```python
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Simple Node has been started')

    def say_hello(self):
        self.get_logger().info('Hello from Simple Node!')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()
    node.say_hello()

    # Keep the node alive
    rclpy.spin(node)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Communication Patterns

Nodes communicate using three main patterns:

1. **Publish/Subscribe (Topics)**: One-to-many communication
2. **Request/Response (Services)**: One-to-one synchronous communication
3. **Action**: One-to-one asynchronous communication with feedback

### Best Practices for Node Design

- Keep nodes focused on a single responsibility
- Use meaningful node names that reflect their function
- Implement proper error handling and logging
- Design nodes to be reusable across different robot platforms
- Consider resource usage and performance when designing nodes

## Topics

Topics are ROS 2's asynchronous communication mechanism that follows a publish/subscribe pattern. They enable one-way data flow from publishers to subscribers and are ideal for continuous data streams like sensor readings, robot states, or command velocities.

### How Topics Work

- **Publishers**: Nodes that send messages to a topic
- **Subscribers**: Nodes that receive messages from a topic
- **Messages**: Data structures that carry information between nodes

The communication is decoupled - publishers don't know who subscribes to their data, and subscribers don't know who publishes the data they receive.

### Creating a Publisher Node

Here's an example of a publisher node that sends string messages:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = TalkerNode()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber Node

Here's the corresponding subscriber that receives messages from the publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = ListenerNode()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topic Characteristics

- **Anonymous**: Publishers and subscribers don't know about each other
- **Asynchronous**: Data is sent without waiting for acknowledgment
- **Many-to-many**: Multiple publishers can publish to the same topic, and multiple subscribers can listen to the same topic
- **Typed**: Each topic has a specific message type that defines its structure

### Common Topic Use Cases

- Sensor data streams (LIDAR, cameras, IMU)
- Robot state information (joint positions, odometry)
- Command velocities for mobile base control
- Robot status and diagnostic information

## Services

Services in ROS 2 provide synchronous request/response communication between nodes. Unlike topics, services establish a direct connection between a client and a server, making them ideal for operations that require immediate responses or acknowledgments.

### How Services Work

- **Service Server**: Provides a specific functionality
- **Service Client**: Requests the service and waits for a response
- **Service Interface**: Defines the request and response message types

### Creating a Service Server

Here's an example of a simple service that adds two integers:

First, define the service interface in a `.srv` file (e.g., `AddTwoInts.srv`):
```
int64 a
int64 b
---
int64 sum
```

Then implement the service server:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    service = AddTwoIntsService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service Client

Here's how to create a client that calls the service:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = AddTwoIntsClient()
    response = client.send_request(2, 3)
    client.get_logger().info(f'Result: {response.sum}')
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### When to Use Services

- Operations that require immediate responses
- Configuration changes that need acknowledgment
- One-time requests that don't require continuous updates
- Operations that have clear success/failure outcomes

### Service vs Topic Comparison

| Aspect | Topic | Service |
|--------|-------|---------|
| Communication | Publish/Subscribe | Request/Response |
| Timing | Asynchronous | Synchronous |
| Connection | Anonymous | Direct |
| Use Case | Continuous data | One-time requests |

## Bridging Python Agents to ROS Controllers using rclpy

The `rclpy` library is the Python client library for ROS 2, enabling Python-based agents and AI systems to communicate with ROS-based robot controllers. This bridge is crucial for integrating modern AI approaches with established robotic frameworks.

### Setting Up rclpy

First, install the required packages:
```bash
pip install rclpy
```

Or ensure ROS 2 is properly sourced in your environment:
```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
```

### Basic rclpy Node Structure

```python
import rclpy
from rclpy.node import Node
import time

class AgentController(Node):
    def __init__(self):
        super().__init__('agent_controller')

        # Initialize publishers, subscribers, services, etc.
        self.get_logger().info('Agent Controller Node Started')

    def agent_behavior(self):
        """Implement your AI agent's behavior here"""
        self.get_logger().info('Executing agent behavior')
        # Your agent logic goes here
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = AgentController()

    # Run agent behavior
    controller.agent_behavior()

    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integrating with Robot Controllers

Here's an example of how to send velocity commands to a robot base controller:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer to send commands at regular intervals
        self.timer = self.create_timer(0.1, self.send_velocity_command)

        # Simple movement pattern
        self.command_index = 0

    def send_velocity_command(self):
        msg = Twist()

        # Simple pattern: move forward, turn, stop
        if self.command_index < 10:
            # Move forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        elif self.command_index < 20:
            # Turn
            msg.linear.x = 0.0
            msg.angular.z = 0.5
        else:
            # Stop and reset
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.command_index = 0

        self.cmd_vel_pub.publish(msg)
        self.command_index += 1
        self.get_logger().info(f'Velocity command: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    controller = VelocityController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Agent Integration Example

Here's a more complex example showing how to create an AI agent that responds to sensor data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class ObstacleAvoidanceAgent(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_agent')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Agent state
        self.safe_distance = 1.0  # meters
        self.linear_speed = 0.3
        self.angular_speed = 0.5

    def scan_callback(self, msg):
        # Process laser scan data to detect obstacles
        ranges = np.array(msg.ranges)
        # Filter out invalid readings (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            return

        min_distance = np.min(valid_ranges)

        # Simple obstacle avoidance behavior
        cmd_msg = Twist()

        if min_distance < self.safe_distance:
            # Obstacle detected - turn to avoid
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = self.angular_speed
            self.get_logger().info(f'Obstacle detected! Distance: {min_distance:.2f}m - Turning')
        else:
            # Path clear - move forward
            cmd_msg.linear.x = self.linear_speed
            cmd_msg.angular.z = 0.0
            self.get_logger().info(f'Path clear. Distance: {min_distance:.2f}m - Moving forward')

        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    agent = ObstacleAvoidanceAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Agent Integration

- **Error Handling**: Always implement proper error handling for ROS communication
- **Resource Management**: Clean up nodes and resources properly
- **Real-time Constraints**: Consider timing requirements for robot control
- **Safety**: Implement safety checks to prevent dangerous robot behavior
- **Modularity**: Design agents as modular components that can be easily tested and replaced

## Understanding URDF (Unified Robot Description Format) for Humanoids

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. For humanoid robots, URDF defines the physical structure, including links (rigid parts), joints (connections between links), and their properties.

### URDF Structure for Humanoids

A humanoid URDF typically includes:

- **Links**: Represent rigid parts like head, torso, arms, legs
- **Joints**: Define how links connect and move relative to each other
- **Materials**: Define visual appearance
- **Inertial properties**: Mass, center of mass, and inertia tensors
- **Visual and collision properties**: Shape, size, and collision detection

### Basic URDF Example

Here's a simplified URDF for a basic humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.15"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting torso to head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>
</robot>
```

### Key URDF Components for Humanoids

#### 1. Links
Links represent rigid bodies in the robot. For humanoids:
- Each body part (head, torso, limbs) is typically a separate link
- Links define visual appearance, collision properties, and inertial properties
- Visual properties determine how the robot appears in simulation
- Collision properties define how the robot interacts with the environment

#### 2. Joints
Joints define the relationship between links:
- **Fixed**: No movement between links (e.g., sensor mounts)
- **Revolute**: Rotational joint with limits (e.g., elbow, knee)
- **Continuous**: Rotational joint without limits (e.g., wheel)
- **Prismatic**: Linear sliding joint (e.g., linear actuators)
- **Floating**: 6-DOF joint (rarely used in humanoids)

#### 3. Materials
Define the visual appearance of links:
- Colors using RGBA values (Red, Green, Blue, Alpha)
- Can reference external texture files

#### 4. Inertial Properties
Critical for physics simulation:
- Mass of each link
- Center of mass location
- Inertia tensor (how mass is distributed)

### URDF Best Practices for Humanoids

1. **Proper Scaling**: Ensure all dimensions are realistic for humanoid proportions
2. **Mass Distribution**: Assign realistic masses based on actual robot hardware
3. **Joint Limits**: Set appropriate limits to prevent damage and ensure realistic movement
4. **Collision Detection**: Design collision geometry that's both accurate and computationally efficient
5. **Inertial Properties**: Use realistic values for proper physics simulation
6. **Gazebo Integration**: Add Gazebo-specific tags if using the Gazebo simulator

### Visualizing URDF

You can visualize URDF files using ROS tools:
- `rviz2`: Real-time visualization of robot models
- `gazebo`: Physics simulation with visualization
- `check_urdf`: Command-line tool to validate URDF syntax

```bash
# Check URDF syntax
check_urdf /path/to/your/robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2
```

## Learning Outcomes

By the end of this module, you should be able to:

### Knowledge
- [ ] Explain the fundamental concepts of ROS 2 and its role as a robotic nervous system
- [ ] Describe the key differences between ROS 1 and ROS 2
- [ ] Understand the publish/subscribe communication pattern in ROS 2 topics
- [ ] Explain the request/response communication pattern in ROS 2 services
- [ ] Define the purpose and structure of URDF files for humanoid robots
- [ ] Identify the components of a ROS 2 node and their functions

### Skills
- [ ] Create and run basic ROS 2 nodes using Python and rclpy
- [ ] Implement publisher and subscriber nodes for topic-based communication
- [ ] Develop service servers and clients for request/response communication
- [ ] Write simple URDF files for basic humanoid robot models
- [ ] Use rclpy to bridge Python-based AI agents with ROS controllers
- [ ] Debug basic ROS 2 communication issues

### Application
- [ ] Design a modular ROS 2 system architecture for a humanoid robot
- [ ] Integrate sensor data processing with robot control using ROS 2 topics
- [ ] Implement basic AI behaviors using ROS 2 communication patterns
- [ ] Create URDF models that accurately represent humanoid robot kinematics
- [ ] Develop Python agents that can control robot behavior through ROS 2 interfaces
- [ ] Apply ROS 2 best practices for distributed robotic systems