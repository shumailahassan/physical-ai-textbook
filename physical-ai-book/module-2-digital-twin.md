# Module 2 — The Digital Twin (Gazebo & Unity)

## Introduction

A digital twin is a virtual representation of a physical robot that exists in a simulated environment. In robotics, digital twins serve as crucial tools for testing, validation, and development without the risks and costs associated with physical hardware. This module explores two primary simulation platforms: Gazebo for physics-based simulation and Unity for high-fidelity visualization and human-robot interaction.

Digital twins enable:
- **Safe testing**: Experiment with robot behaviors without risk of hardware damage
- **Rapid prototyping**: Quickly iterate on robot designs and algorithms
- **Training**: Develop and test AI agents in controlled environments
- **Validation**: Verify robot performance before deployment
- **Cost reduction**: Minimize hardware requirements during development

The combination of Gazebo and Unity provides a comprehensive simulation environment where physics accuracy meets visual fidelity, creating an ideal platform for developing and testing humanoid robots.

## Physics Simulation in Gazebo

Gazebo is a powerful physics simulator that provides realistic simulation of robots in complex environments. It uses advanced physics engines like ODE (Open Dynamics Engine), Bullet, and DART to accurately model the physical interactions between robots, objects, and the environment.

### Core Physics Concepts in Gazebo

Gazebo's physics simulation is built on several fundamental principles:

- **Rigid Body Dynamics**: Each object is treated as a rigid body with mass, center of mass, and inertia properties
- **Collision Detection**: Algorithms detect when objects intersect and compute appropriate responses
- **Force Application**: Gravity, friction, and user-defined forces affect object motion
- **Joint Constraints**: Limit the degrees of freedom between connected links

### Setting Up a Gazebo Simulation

Here's a basic example of launching a Gazebo simulation with a robot model:

```xml
<!-- launch_simulation.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # Launch Gazebo with a world file
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'simple_room.world'
            ])
        }.items()
    )

    ld.add_action(gazebo)

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )

    ld.add_action(spawn_entity)
    return ld
```

### Physics Parameters and Tuning

To achieve realistic simulation, several parameters need careful tuning:

- **Gravity**: Typically set to -9.81 m/s² on Earth
- **Friction coefficients**: Static and dynamic friction values for realistic contact
- **Damping**: Linear and angular damping to simulate energy loss
- **Solver parameters**: Step size, iterations, and error correction settings

```xml
<!-- Physics configuration in world file -->
<sdf version="1.7">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Rest of the world definition -->
  </world>
</sdf>
```

### Gazebo Plugins for Enhanced Physics

Gazebo supports plugins that extend its capabilities:

- **Joint Control Plugins**: Enable precise control of robot joints
- **Sensor Plugins**: Simulate various sensor types with realistic noise models
- **Model Plugins**: Add custom behaviors to robot models
- **World Plugins**: Modify global simulation parameters

### Common Physics Challenges and Solutions

- **Stability**: Use smaller time steps for more stable simulation
- **Penetration**: Increase constraint iterations to reduce object penetration
- **Performance**: Balance accuracy with computational efficiency
- **Realism**: Tune parameters to match real-world behavior

## Environment Building

Creating realistic environments is crucial for effective robot simulation. The environment determines how robots interact with their surroundings and affects the validity of simulation results.

### Types of Environments

#### Indoor Environments
- **Homes**: Furniture, narrow passages, stairs
- **Offices**: Desks, chairs, doorways, elevators
- **Warehouses**: Shelves, pallets, large open spaces
- **Hospitals**: Corridors, medical equipment, sterile requirements

#### Outdoor Environments
- **Urban**: Roads, sidewalks, traffic, buildings
- **Natural**: Terrain, vegetation, weather conditions
- **Industrial**: Construction sites, factories, heavy machinery

### Building Tools and Workflows

#### Model Creation Process
1. **Conceptual Design**: Sketch the environment layout
2. **3D Modeling**: Create geometric representations
3. **Material Assignment**: Apply visual and physical properties
4. **Lighting Setup**: Configure illumination
5. **Testing and Iteration**: Validate in simulation

#### SDF (Simulation Description Format)
SDF is the XML-based format used by Gazebo to describe simulation environments:

```xml
<!-- Example SDF world file -->
<sdf version="1.7">
  <world name="simple_room">
    <!-- Include models from Gazebo Model Database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom furniture -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Wall -->
    <model name="wall_1">
      <pose>0 -3 1 0 0 0</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Environment Design Best Practices

#### Performance Optimization
- **Level of Detail (LOD)**: Reduce complexity for distant objects
- **Occlusion**: Hide objects not visible to sensors
- **Instancing**: Reuse similar objects to save memory
- **Collision Simplification**: Use simpler shapes for collision detection

#### Realism vs. Performance Trade-offs
- **High-fidelity visuals**: Better for human-robot interaction studies
- **Simplified physics**: Faster for algorithm development
- **Procedural generation**: Efficient for large environments
- **Modular design**: Easy to modify and extend

### Advanced Environment Features

#### Dynamic Elements
- **Moving obstacles**: Simulate pedestrians, vehicles, or moving furniture
- **Interactive objects**: Items that robots can manipulate
- **Changing conditions**: Moving doors, elevators, or adjustable lighting

#### Environmental Effects
- **Weather simulation**: Rain, snow, fog impact on sensors
- **Lighting changes**: Day/night cycles, shadows, reflections
- **Acoustic properties**: Sound propagation and absorption

## High-fidelity Rendering and Human-Robot Interaction in Unity

Unity provides high-quality 3D rendering capabilities that complement Gazebo's physics simulation. While Gazebo focuses on accurate physics, Unity excels at visual realism and human-robot interaction scenarios.

### Unity Integration with Robotics

Unity's capabilities make it ideal for:
- **Visual realism**: High-quality graphics and lighting
- **Human-robot interaction**: Intuitive interfaces for human operators
- **VR/AR applications**: Immersive teleoperation and training
- **Data visualization**: Complex information presentation
- **User experience testing**: Evaluating robot interfaces

### Setting up Unity for Robotics

Unity can interface with ROS 2 through several methods:

1. **Unity Robotics Hub**: Official Unity package for ROS integration
2. **ROS#**: .NET-based ROS bridge for Unity
3. **Custom TCP/IP communication**: Direct socket-based communication

Here's an example setup using Unity Robotics Hub:

```csharp
// RobotController.cs - Basic Unity robot controller
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/cmd_vel";

    private ROSConnection ros;
    private Vector3 targetVelocity = Vector3.zero;

    void Start()
    {
        ros = ROSConnection.instance;
        // Subscribe to robot state topics if needed
    }

    void Update()
    {
        // Update robot position based on target velocity
        transform.Translate(targetVelocity * Time.deltaTime);
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        // Create Twist message
        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(linearX, 0, 0);
        twist.angular = new Vector3Msg(0, 0, angularZ);

        // Send to ROS
        ros.Send<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TwistMsg>(topicName, twist);
    }
}
```

### Visual Quality and Rendering

Unity's rendering pipeline supports:

- **Physically Based Rendering (PBR)**: Realistic material properties
- **Global Illumination**: Accurate light bouncing and shadows
- **Post-processing effects**: Bloom, depth of field, color grading
- **Real-time ray tracing**: Advanced lighting effects (if hardware supports)

### Human-Robot Interaction Scenarios

#### Teleoperation Interfaces
- **First-person view**: Operator sees through robot's cameras
- **Third-person view**: Observer perspective of robot in environment
- **Augmented reality overlays**: Additional information on camera feeds
- **Gesture recognition**: Hand tracking for intuitive control

#### Training and Evaluation
- **Immersive environments**: VR headsets for realistic training
- **Scenario replay**: Review and analyze robot behaviors
- **Multi-user collaboration**: Multiple operators working together
- **Performance metrics**: Real-time feedback on robot performance

### Unity Asset Creation for Robotics

Creating custom assets for robotic applications:

```csharp
// SensorVisualization.cs - Visualize sensor data in Unity
using UnityEngine;

public class SensorVisualization : MonoBehaviour
{
    [SerializeField] private LineRenderer lineRenderer;
    [SerializeField] private int maxPoints = 1000;

    public void UpdateLidarVisualization(float[] ranges, float angleIncrement, Vector3 robotPosition)
    {
        if (lineRenderer == null) return;

        // Calculate number of points to display
        int pointCount = Mathf.Min(ranges.Length, maxPoints);
        lineRenderer.positionCount = pointCount;

        for (int i = 0; i < pointCount; i++)
        {
            float angle = i * angleIncrement - (ranges.Length * angleIncrement / 2);
            float distance = ranges[i];

            Vector3 point = new Vector3(
                robotPosition.x + distance * Mathf.Cos(angle),
                robotPosition.y,
                robotPosition.z + distance * Mathf.Sin(angle)
            );

            lineRenderer.SetPosition(i, point);
        }
    }
}
```

### Performance Considerations

- **LOD Systems**: Reduce detail for distant objects
- **Occlusion Culling**: Hide objects not visible to cameras
- **Texture Streaming**: Load textures on-demand
- **Shader Optimization**: Use efficient shaders for real-time rendering
- **Multi-threading**: Offload computation to background threads

## Simulating Sensors: LiDAR, Depth Cameras, and IMUs

Accurate sensor simulation is critical for developing robust robot perception and navigation systems. This section covers the simulation of three essential sensor types.

### LiDAR Simulation

LiDAR (Light Detection and Ranging) sensors provide 2D or 3D range measurements by emitting laser pulses and measuring the time of flight of reflected light.

#### LiDAR in Gazebo

Gazebo provides realistic LiDAR simulation through its sensor plugins:

```xml
<!-- LiDAR sensor definition in URDF/XACRO -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π -->
          <max_angle>3.14159</max_angle>    <!-- π -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

#### LiDAR Data Processing in ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def lidar_callback(self, msg):
        # Convert to numpy array for easier processing
        ranges = np.array(msg.ranges)

        # Filter out invalid readings (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        # Calculate statistics
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            avg_distance = np.mean(valid_ranges)

            self.get_logger().info(
                f'Lidar: min={min_distance:.2f}m, avg={avg_distance:.2f}m, '
                f'valid_points={len(valid_ranges)}/{len(ranges)}'
            )

        # Obstacle detection
        safe_distance = 1.0  # meters
        if len(valid_ranges) > 0 and min_distance < safe_distance:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m!')

def main(args=None):
    rclpy.init(args=args)
    processor = LidarProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### LiDAR Simulation Parameters

Key parameters that affect LiDAR simulation quality:
- **Range resolution**: Accuracy of distance measurements
- **Angular resolution**: Precision of angle measurements
- **Field of view**: Angular coverage of the sensor
- **Update rate**: Frequency of measurements
- **Noise models**: Realistic error simulation
- **Ray count**: Number of beams for 3D LiDAR

### Depth Camera Simulation

Depth cameras provide 2D intensity images along with depth information for each pixel, enabling 3D scene reconstruction and object recognition.

#### Depth Camera in Gazebo

```xml
<!-- Depth camera definition -->
<gazebo reference="camera_link">
  <sensor type="depth" name="depth_camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="depth_cam">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

#### Processing Depth Camera Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')
        self.bridge = CvBridge()

        # Subscribe to depth image
        self.depth_subscription = self.create_subscription(
            Image,
            '/depth_camera/depth/image_raw',
            self.depth_image_callback,
            10
        )

        # Subscribe to RGB image (if available)
        self.rgb_subscription = self.create_subscription(
            Image,
            '/depth_camera/image_raw',
            self.rgb_image_callback,
            10
        )

    def depth_image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Could not convert depth image: {str(e)}')
            return

        # Process depth data
        depth_array = np.array(cv_image, dtype=np.float32)

        # Calculate statistics
        valid_depths = depth_array[np.isfinite(depth_array) & (depth_array > 0)]
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)

            self.get_logger().info(
                f'Depth Camera: avg={avg_depth:.2f}m, '
                f'min={min_depth:.2f}m, max={max_depth:.2f}m'
            )

        # Find objects at specific distance ranges
        close_objects = (valid_depths > 0) & (valid_depths < 1.0)
        if np.sum(close_objects) > 100:  # More than 100 pixels
            self.get_logger().info('Close object detected!')

    def rgb_image_callback(self, msg):
        # Process RGB image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert RGB image: {str(e)}')
            return

        # Example: Convert to grayscale and detect edges
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Display results (in simulation environment)
        cv2.imshow('RGB Image', cv_image)
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    processor = DepthCameraProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### IMU Simulation

An Inertial Measurement Unit (IMU) provides measurements of linear acceleration, angular velocity, and sometimes magnetic field orientation, which are essential for robot localization and balance control.

#### IMU in Gazebo

```xml
<!-- IMU sensor definition -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

#### Processing IMU Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np
import math

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # For orientation estimation
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w (quaternion)

    def imu_callback(self, msg):
        # Extract linear acceleration
        linear_acc = msg.linear_acceleration
        acc_magnitude = math.sqrt(
            linear_acc.x**2 + linear_acc.y**2 + linear_acc.z**2
        )

        # Extract angular velocity
        angular_vel = msg.angular_velocity
        ang_vel_magnitude = math.sqrt(
            angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2
        )

        # Extract orientation (if available)
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        self.get_logger().info(
            f'IMU - Acc: mag={acc_magnitude:.3f}, '
            f'AngVel: mag={ang_vel_magnitude:.3f}, '
            f'Orientation: R={math.degrees(roll):.1f}°, '
            f'P={math.degrees(pitch):.1f}°, Y={math.degrees(yaw):.1f}°'
        )

        # Detect significant movements
        if acc_magnitude > 15.0:  # Threshold for significant acceleration
            self.get_logger().warn('Significant acceleration detected!')

        if ang_vel_magnitude > 1.0:  # Threshold for significant rotation
            self.get_logger().info('Robot is rotating')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Fusion and Integration

Combining data from multiple sensors improves perception accuracy:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist
import numpy as np

class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Subscriptions to all sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        # Publisher for fused decisions
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Sensor data storage
        self.lidar_data = None
        self.imu_data = None
        self.camera_data = None

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)

        # Process LiDAR data for obstacle detection
        if self.lidar_data is not None:
            valid_ranges = self.lidar_data[np.isfinite(self.lidar_data)]
            if len(valid_ranges) > 0 and np.min(valid_ranges) < 1.0:
                self.avoid_obstacle()

    def imu_callback(self, msg):
        self.imu_data = msg
        # Process IMU data for stability
        linear_acc = msg.linear_acceleration
        total_acc = np.sqrt(linear_acc.x**2 + linear_acc.y**2 + linear_acc.z**2)

        if total_acc > 15.0:  # Potential fall or impact
            self.get_logger().warn('Potential robot instability detected!')

    def camera_callback(self, msg):
        self.camera_data = msg
        # Process camera data for object detection
        # (Implementation would include computer vision algorithms)

    def avoid_obstacle(self):
        """Implement obstacle avoidance behavior"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.5  # Turn to avoid obstacle
        self.cmd_pub.publish(cmd_msg)
        self.get_logger().info('Avoiding obstacle based on sensor fusion')

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusion()
    rclpy.spin(fusion_node)
    fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Learning Outcomes

By the end of this module, you should be able to:

### Knowledge
- [ ] Explain the concept and importance of digital twins in robotics
- [ ] Understand the differences between Gazebo and Unity for simulation
- [ ] Describe the physics simulation capabilities of Gazebo
- [ ] Identify key parameters that affect simulation accuracy
- [ ] Explain how to build realistic environments for robot testing
- [ ] Understand the principles of sensor simulation for LiDAR, cameras, and IMUs
- [ ] Describe how Unity enhances human-robot interaction scenarios

### Skills
- [ ] Create and configure Gazebo simulation environments
- [ ] Implement physics-based robot models in Gazebo
- [ ] Build custom environments with appropriate complexity
- [ ] Integrate Unity with ROS 2 for high-fidelity visualization
- [ ] Simulate and process data from LiDAR sensors
- [ ] Simulate and process data from depth cameras
- [ ] Simulate and process data from IMU sensors
- [ ] Implement sensor fusion techniques in simulation

### Application
- [ ] Design simulation scenarios that match real-world requirements
- [ ] Create physically accurate models of robots and environments
- [ ] Develop and test robot behaviors in simulated environments
- [ ] Validate sensor processing algorithms using simulated data
- [ ] Implement human-robot interaction interfaces using Unity
- [ ] Evaluate robot performance through simulation-based testing
- [ ] Transition successfully from simulation to real hardware
- [ ] Optimize simulation parameters for performance and accuracy balance