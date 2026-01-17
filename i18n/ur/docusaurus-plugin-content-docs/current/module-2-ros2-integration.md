---
# Translation needed
# اس صفحے کا ترجمہ درکار ہے
---

**This page is currently only available in English. Please contribute a translation.**
**یہ صفحہ فی الحال صرف انگریزی میں دستیاب ہے۔ براہ کرم ایک ترجمہ شامل کریں۔**

---
id: module-2-ros2-integration
title: Chapter 7 - Integration with ROS2
sidebar_label: Chapter 7 - Integration with ROS2
---

# Chapter 7: Integration with ROS2

## ROS2-Gazebo Bridge

The integration between ROS2 and Gazebo forms the backbone of many robotics simulation workflows. This bridge enables seamless communication between the ROS2 middleware and the Gazebo physics simulator, allowing for sophisticated robot simulation and control.

### ROS2-Gazebo Integration Architecture

The ROS2-Gazebo integration relies on the `gazebo_ros_pkgs` package, which provides plugins and launch files to connect the two systems. The architecture consists of:

- **Gazebo Server**: Handles physics simulation and sensor data generation
- **Gazebo Client**: Provides visualization interface
- **ROS2 Bridge**: Facilitates communication between Gazebo and ROS2
- **ROS2 Nodes**: Handle robot control, perception, and other high-level tasks

### gazebo_ros_pkgs and Available Plugins

The `gazebo_ros_pkgs` package provides essential plugins for ROS2-Gazebo integration:

#### Core Plugins

```xml
<!-- Example: Complete robot configuration with ROS2 integration -->
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include ROS2 control plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Joint state publisher plugin -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <update_rate>30</update_rate>
      <joint_name>hip_joint</joint_name>
      <joint_name>knee_joint</joint_name>
      <joint_name>ankle_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Diff drive controller plugin (for wheeled robots) -->
  <gazebo>
    <plugin name="diff_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <robotBaseFrame>base_link</robotBaseFrame>
      <updateRate>30</updateRate>
      <leftJoint>left_wheel_joint</leftJoint>
      <rightJoint>right_wheel_joint</rightJoint>
      <wheelSeparation>0.3</wheelSeparation>
      <wheelDiameter>0.15</wheelDiameter>
    </plugin>
  </gazebo>

  <!-- Joint trajectory controller -->
  <gazebo>
    <plugin name="joint_trajectory_controller" filename="libgazebo_ros_joint_trajectory.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <topicName>joint_trajectory</topicName>
      <updateRate>100</updateRate>
    </plugin>
  </gazebo>

</robot>
```

### Communication Protocols and Message Types

The ROS2-Gazebo bridge uses standard ROS2 message types for communication:

```cpp
// Example: ROS2 node for controlling a simulated humanoid robot in Gazebo
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>

class GazeboRobotController : public rclcpp::Node
{
public:
    GazeboRobotController()
    : Node("gazebo_robot_controller")
    {
        // Subscribe to joint states from Gazebo
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&GazeboRobotController::jointStateCallback, this, std::placeholders::_1));

        // Publisher for joint commands
        joint_cmd_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory", 10);

        // Publisher for velocity commands
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), // 100 Hz
            std::bind(&GazeboRobotController::controlLoop, this));
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Store current joint states
        current_joint_states_ = *msg;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Received joint states: %zu joints", msg->name.size());
    }

    void controlLoop()
    {
        // Implement control logic based on current state
        if (!current_joint_states_.name.empty()) {
            // Example: Send a simple joint trajectory
            sendJointTrajectoryCommand();
        }
    }

    void sendJointTrajectoryCommand()
    {
        auto trajectory_msg = trajectory_msgs::msg::JointTrajectory();
        trajectory_msg.joint_names = {"hip_joint", "knee_joint", "ankle_joint"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {0.1, -0.5, 0.2};  // Desired joint positions
        point.velocities = {0.0, 0.0, 0.0};  // Desired joint velocities
        point.time_from_start = rclcpp::Duration::from_seconds(0.1);

        trajectory_msg.points.push_back(point);

        joint_cmd_pub_->publish(trajectory_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_cmd_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    sensor_msgs::msg::JointState current_joint_states_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GazeboRobotController>());
    rclcpp::shutdown();
    return 0;
}
```

### Best Practices for Integration

When integrating ROS2 with Gazebo, consider these best practices:

1. **Proper Namespace Management**: Use consistent namespaces across all topics and parameters
2. **Update Rate Optimization**: Balance simulation accuracy with computational efficiency
3. **Error Handling**: Implement robust error handling for connection issues
4. **Resource Management**: Monitor CPU and memory usage during simulation

## ROS2-Unity Bridge

The integration between ROS2 and Unity enables sophisticated simulation environments with high-quality graphics and physics. This bridge is typically achieved using the Unity ROS TCP Connector package.

### ROS2-Unity Integration Approaches

The Unity-ROS2 bridge is primarily implemented through the Unity Robotics Hub, which includes:

- **ROS TCP Connector**: Enables communication between Unity and ROS2 via TCP
- **URDF Importer**: Allows direct import of ROS robot models into Unity
- **Message Packages**: Support for common ROS message types in Unity

### Available Unity-ROS2 Packages

#### ROS TCP Connector Setup

```csharp
// Example: Unity ROS2 bridge setup
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("ROS2 Configuration")]
    public string rosIpAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string trajectoryTopic = "/joint_trajectory";

    private ROSConnection ros;
    private bool isConnected = false;

    void Start()
    {
        // Connect to ROS2
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIpAddress, rosPort);

        // Subscribe to topics
        ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);

        // Connection status check
        InvokeRepeating("CheckConnection", 1.0f, 5.0f);
    }

    void CheckConnection()
    {
        if (ros != null && ros.IsConnected())
        {
            isConnected = true;
            Debug.Log("Connected to ROS2");
        }
        else
        {
            isConnected = false;
            Debug.LogWarning("Not connected to ROS2");
        }
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Process joint state data
        Debug.Log($"Received joint state with {jointState.name.Length} joints");

        // Update Unity robot model based on joint positions
        UpdateRobotModel(jointState);
    }

    void UpdateRobotModel(JointStateMsg jointState)
    {
        // Update each joint in the Unity robot model
        for (int i = 0; i < jointState.name.Length && i < jointState.position.Length; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = jointState.position[i];

            // Find and update the corresponding joint in Unity
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Apply rotation based on joint type
                jointTransform.localRotation = Quaternion.Euler(0, 0, jointPosition * Mathf.Rad2Deg);
            }
        }
    }

    Transform FindJointByName(string name)
    {
        // Find joint by name in the robot hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (!isConnected) return;

        var twistMsg = new TwistMsg
        {
            linear = new Vector3Msg { x = linearX, y = 0, z = 0 },
            angular = new Vector3Msg { x = 0, y = 0, z = angularZ }
        };

        ros.Publish(cmdVelTopic, twistMsg);
    }

    public void SendJointTrajectory(string[] jointNames, float[] positions, float[] velocities)
    {
        if (!isConnected) return;

        var trajectoryMsg = new JointTrajectoryMsg
        {
            joint_names = jointNames,
            points = new JointTrajectoryPointMsg[1]
        };

        trajectoryMsg.points[0] = new JointTrajectoryPointMsg
        {
            positions = positions,
            velocities = velocities,
            time_from_start = new DurationMsg { sec = 1, nanosec = 0 }
        };

        ros.Publish(trajectoryTopic, trajectoryMsg);
    }
}
```

### Communication Protocols and Message Types

Unity-ROS2 communication follows standard ROS2 message types but uses TCP-based transport:

```csharp
// Example: Custom ROS2 message handler for Unity
using System;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnitySensorManager : MonoBehaviour
{
    [Header("Sensor Topics")]
    public string imuTopic = "/imu/data";
    public string laserScanTopic = "/scan";
    public string cameraTopic = "/camera/image_raw";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to sensor topics
        ros.Subscribe<ImuMsg>(imuTopic, OnImuDataReceived);
        ros.Subscribe<LaserScanMsg>(laserScanTopic, OnLaserScanReceived);
        ros.Subscribe<ImageMsg>(cameraTopic, OnCameraReceived);
    }

    void OnImuDataReceived(ImuMsg imuData)
    {
        // Process IMU data
        Vector3 orientation = new Vector3(
            (float)imuData.orientation.x,
            (float)imuData.orientation.y,
            (float)imuData.orientation.z
        );

        // Update Unity object orientation based on IMU data
        transform.rotation = Quaternion.Euler(orientation);
    }

    void OnLaserScanReceived(LaserScanMsg laserScan)
    {
        // Process laser scan data for visualization or collision detection
        float[] ranges = laserScan.ranges;

        // Example: Create visualization of laser scan
        VisualizeLaserScan(ranges, laserScan.angle_min, laserScan.angle_increment);
    }

    void VisualizeLaserScan(float[] ranges, float angleMin, float angleIncrement)
    {
        // Create visualization of laser scan points
        for (int i = 0; i < ranges.Length; i++)
        {
            float angle = angleMin + (i * angleIncrement);
            float distance = ranges[i];

            if (distance > laserScan.range_min && distance < laserScan.range_max)
            {
                Vector3 point = new Vector3(
                    distance * Mathf.Cos(angle),
                    0,
                    distance * Mathf.Sin(angle)
                );

                // Visualize the point (e.g., with a small sphere)
                // Instantiate visualization prefab at 'point'
            }
        }
    }

    void OnCameraReceived(ImageMsg imageData)
    {
        // Process camera image data
        // This would typically involve updating a Unity texture with the image data
        UpdateCameraTexture(imageData);
    }

    void UpdateCameraTexture(ImageMsg imageMsg)
    {
        // Convert ROS image message to Unity texture
        // Implementation would depend on image format and requirements
    }
}
```

### Best Practices for Unity-ROS2 Integration

1. **Performance Optimization**: Use appropriate update rates to balance real-time performance with simulation accuracy
2. **Network Configuration**: Ensure proper network setup for reliable TCP communication
3. **Message Serialization**: Optimize message sizes for efficient transmission
4. **Error Handling**: Implement robust connection management and reconnection logic

## Sensor Integration with ROS2

Integrating simulated sensors with ROS2 involves publishing sensor data to appropriate ROS2 topics and ensuring proper message formats.

### Connecting Simulated Sensors to ROS2

#### Gazebo Sensor Integration

```xml
<!-- Example: Complete sensor configuration for ROS2 integration -->
<!-- IMU Sensor -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topicName>/imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <serviceName>/imu/service</serviceName>
      <gaussianNoise>0.0017</gaussianNoise>
      <updateRateHZ>100.0</updateRateHZ>
      <frameName>imu_link</frameName>
    </plugin>
  </sensor>
</gazebo>

<!-- Camera Sensor -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>camera</cameraName>
      <imageTopicName>/camera/image_raw</imageTopicName>
      <cameraInfoTopicName>/camera/camera_info</cameraInfoTopicName>
      <frameName>camera_link</frameName>
      <hackBaseline>0.07</hackBaseline>
      <distortionK1>0.0</distortionK1>
      <distortionK2>0.0</distortionK2>
      <distortionK3>0.0</distortionK3>
      <distortionT1>0.0</distortionT1>
      <distortionT2>0.0</distortionT2>
    </plugin>
  </sensor>
</gazebo>

<!-- LIDAR Sensor -->
<gazebo reference="lidar_link">
  <sensor name="lidar_2d" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topicName>/scan</topicName>
      <frameName>lidar_link</frameName>
      <minRange>0.1</minRange>
      <maxRange>30.0</maxRange>
      <updateRate>10.0</updateRate>
    </plugin>
  </sensor>
</gazebo>
```

#### Unity Sensor Integration

```csharp
// Example: Unity sensor data publisher for ROS2
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnitySensorPublisher : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public float updateRate = 30.0f;

    [Header("ROS2 Topics")]
    public string jointStatesTopic = "/joint_states";
    public string imuTopic = "/imu/data";
    public string laserScanTopic = "/scan";

    private ROSConnection ros;
    private float nextUpdateTime;
    private JointStateMsg jointStateMsg;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        nextUpdateTime = Time.time;

        InitializeJointStateMessage();
    }

    void InitializeJointStateMessage()
    {
        // Find all joints in the robot
        var jointComponents = GetComponentsInChildren<HumanoidJoint>();

        jointStateMsg = new JointStateMsg();
        jointStateMsg.name = new string[jointComponents.Length];
        jointStateMsg.position = new double[jointComponents.Length];
        jointStateMsg.velocity = new double[jointComponents.Length];
        jointStateMsg.effort = new double[jointComponents.Length];

        for (int i = 0; i < jointComponents.Length; i++)
        {
            jointStateMsg.name[i] = jointComponents[i].jointName;
        }
    }

    void Update()
    {
        if (Time.time >= nextUpdateTime)
        {
            PublishSensorData();
            nextUpdateTime = Time.time + (1.0f / updateRate);
        }
    }

    void PublishSensorData()
    {
        // Publish joint states
        UpdateJointStates();
        ros.Publish(jointStatesTopic, jointStateMsg);

        // Publish IMU data if available
        var imuComponent = GetComponent<UnityIMUSensor>();
        if (imuComponent != null)
        {
            var imuData = imuComponent.GetIMUData();
            ros.Publish(imuTopic, imuData);
        }

        // Publish LIDAR data if available
        var lidarComponent = GetComponent<UnityLIDARSensor>();
        if (lidarComponent != null)
        {
            var lidarData = lidarComponent.GetLIDARData();
            ros.Publish(laserScanTopic, lidarData);
        }
    }

    void UpdateJointStates()
    {
        var jointComponents = GetComponentsInChildren<HumanoidJoint>();

        for (int i = 0; i < jointComponents.Length && i < jointStateMsg.name.Length; i++)
        {
            jointStateMsg.position[i] = jointComponents[i].GetCurrentAngle();
            jointStateMsg.velocity[i] = jointComponents[i].GetCurrentVelocity();
            jointStateMsg.effort[i] = jointComponents[i].GetCurrentEffort();
        }

        // Set timestamp
        jointStateMsg.header = new HeaderMsg
        {
            stamp = new TimeMsg
            {
                sec = (int)Time.time,
                nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9f)
            },
            frame_id = "base_link"
        };
    }
}

// Helper component for individual joints
public class HumanoidJoint : MonoBehaviour
{
    public string jointName;
    public JointType jointType;
    public float minAngle = -90f;
    public float maxAngle = 90f;

    private float currentAngle = 0f;
    private float currentVelocity = 0f;
    private float currentEffort = 0f;

    public float GetCurrentAngle()
    {
        // Get current joint angle based on transform
        switch (jointType)
        {
            case JointType.Revolute:
                return transform.localEulerAngles.z * Mathf.Deg2Rad;
            case JointType.Prismatic:
                return transform.localPosition.x; // or appropriate axis
            default:
                return 0f;
        }
    }

    public float GetCurrentVelocity()
    {
        // Calculate or store velocity
        return currentVelocity;
    }

    public float GetCurrentEffort()
    {
        // Return current effort/torque
        return currentEffort;
    }
}

public enum JointType
{
    Revolute,
    Prismatic,
    Fixed,
    Continuous
}
```

### Message Publishing from Simulation

The process of publishing sensor data from simulation to ROS2 involves:

1. **Data Collection**: Gather sensor data from the simulation environment
2. **Message Construction**: Format the data according to ROS2 message standards
3. **Topic Publishing**: Publish the formatted messages to appropriate ROS2 topics
4. **Timing Management**: Ensure appropriate update rates for real-time performance

## Control Integration with ROS2

Control integration involves sending commands from ROS2 nodes to the simulation environment to control robot behavior.

### Sending Control Commands to Simulation

#### Gazebo Control Integration

```xml
<!-- Example: ROS2 control interface configuration -->
<!-- Joint state controller -->
<gazebo>
  <plugin name="joint_state_controller" filename="libgazebo_ros_joint_state_publisher.so">
    <update_rate>30</update_rate>
    <joint_name>hip_joint</joint_name>
    <joint_name>knee_joint</joint_name>
    <joint_name>ankle_joint</joint_name>
  </plugin>
</gazebo>

<!-- Position joint controller -->
<gazebo>
  <plugin name="position_joint_controller" filename="libgazebo_ros_joint_pose.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <serviceName>set_joint_positions</serviceName>
    <commandTopic>joint_position_command</commandTopic>
    <stateTopic>joint_position_state</stateTopic>
  </plugin>
</gazebo>

<!-- Velocity joint controller -->
<gazebo>
  <plugin name="velocity_joint_controller" filename="libgazebo_ros_joint_trajectory.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <commandTopic>joint_trajectory</commandTopic>
    <stateTopic>joint_trajectory_state</stateTopic>
  </plugin>
</gazebo>
```

#### Control Node Implementation

```cpp
// Example: ROS2 control node for simulation
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class SimulationController : public rclcpp::Node
{
public:
    SimulationController()
    : Node("simulation_controller")
    {
        // Subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&SimulationController::jointStateCallback, this, std::placeholders::_1));

        // Publishers
        joint_trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory", 10);

        joint_command_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/position_commands", 10);

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), // 100 Hz
            std::bind(&SimulationController::controlLoop, this));

        RCLCPP_INFO(this->get_logger(), "Simulation controller initialized");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        current_joint_states_ = *msg;
    }

    void controlLoop()
    {
        // Example: Simple PD controller for maintaining a target position
        if (current_joint_states_.name.empty()) {
            return;
        }

        // Define target positions
        std::vector<double> target_positions = {0.1, -0.5, 0.2}; // hip, knee, ankle
        std::vector<double> current_positions;

        // Extract current positions
        for (const auto& joint_name : joint_names_) {
            auto it = std::find(current_joint_states_.name.begin(),
                              current_joint_states_.name.end(), joint_name);
            if (it != current_joint_states_.name.end()) {
                size_t index = std::distance(current_joint_states_.name.begin(), it);
                if (index < current_joint_states_.position.size()) {
                    current_positions.push_back(current_joint_states_.position[index]);
                }
            }
        }

        if (current_positions.size() == target_positions.size()) {
            // Calculate control commands
            std::vector<double> commands = computePDControl(target_positions, current_positions);

            // Publish position commands
            publishPositionCommands(commands);
        }
    }

    std::vector<double> computePDControl(const std::vector<double>& targets,
                                       const std::vector<double>& current)
    {
        std::vector<double> commands;
        for (size_t i = 0; i < targets.size(); ++i) {
            double error = targets[i] - current[i];
            double command = kp_ * error; // Simple P control
            commands.push_back(command);
        }
        return commands;
    }

    void publishPositionCommands(const std::vector<double>& commands)
    {
        auto msg = std_msgs::msg::Float64MultiArray();
        msg.data.resize(commands.size());
        std::copy(commands.begin(), commands.end(), msg.data.begin());

        joint_command_pub_->publish(msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    sensor_msgs::msg::JointState current_joint_states_;
    std::vector<std::string> joint_names_ = {"hip_joint", "knee_joint", "ankle_joint"};
    double kp_ = 10.0; // Proportional gain
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimulationController>());
    rclcpp::shutdown();
    return 0;
}
```

#### Unity Control Integration

```csharp
// Example: Unity control receiver for ROS2 commands
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityControlReceiver : MonoBehaviour
{
    [Header("Control Topics")]
    public string jointTrajectoryTopic = "/joint_trajectory";
    public string jointCommandTopic = "/position_commands";
    public string cmdVelTopic = "/cmd_vel";

    private ROSConnection ros;
    private HumanoidRobotController robotController;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        robotController = GetComponent<HumanoidRobotController>();

        // Subscribe to control topics
        ros.Subscribe<JointTrajectoryMsg>(jointTrajectoryTopic, OnJointTrajectoryReceived);
        ros.Subscribe<Float64MultiArrayMsg>(jointCommandTopic, OnJointCommandReceived);
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnVelocityCommandReceived);
    }

    void OnJointTrajectoryReceived(JointTrajectoryMsg trajectory)
    {
        if (trajectory.points.Length > 0)
        {
            var targetPoint = trajectory.points[0];

            // Send joint positions to robot controller
            if (robotController != null)
            {
                robotController.SetJointPositions(targetPoint.positions);
            }
        }
    }

    void OnJointCommandReceived(Float64MultiArrayMsg command)
    {
        // Direct position commands
        if (robotController != null)
        {
            double[] positions = command.data;
            float[] floatPositions = new float[positions.Length];

            for (int i = 0; i < positions.Length; i++)
            {
                floatPositions[i] = (float)positions[i];
            }

            robotController.SetJointPositions(floatPositions);
        }
    }

    void OnVelocityCommandReceived(TwistMsg twist)
    {
        // Handle velocity commands for mobile base
        if (robotController != null)
        {
            robotController.SetVelocityCommand(
                (float)twist.linear.x,
                (float)twist.linear.y,
                (float)twist.linear.z,
                (float)twist.angular.x,
                (float)twist.angular.y,
                (float)twist.angular.z
            );
        }
    }
}

// Enhanced robot controller with ROS2 integration
public class HumanoidRobotController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public HumanoidJoint[] joints;

    [Header("Control Parameters")]
    public float positionKp = 10.0f;
    public float velocityKp = 5.0f;
    public float maxVelocity = 2.0f;

    private float[] targetPositions;
    private float[] currentPositions;

    void Start()
    {
        InitializeJoints();
        targetPositions = new float[joints.Length];
        currentPositions = new float[joints.Length];
    }

    void InitializeJoints()
    {
        // Find all joint components
        joints = GetComponentsInChildren<HumanoidJoint>();
    }

    void Update()
    {
        UpdateControl();
    }

    void UpdateControl()
    {
        // Update current positions
        for (int i = 0; i < joints.Length; i++)
        {
            currentPositions[i] = joints[i].GetCurrentAngle();
        }

        // Apply control to reach target positions
        ApplyPositionControl();
    }

    void ApplyPositionControl()
    {
        for (int i = 0; i < joints.Length && i < targetPositions.Length; i++)
        {
            float error = targetPositions[i] - currentPositions[i];
            float command = positionKp * error;

            // Apply velocity limit
            command = Mathf.Clamp(command, -maxVelocity, maxVelocity);

            // Apply command to joint
            joints[i].ApplyVelocityCommand(command);
        }
    }

    public void SetJointPositions(float[] positions)
    {
        // Validate input
        if (positions.Length != targetPositions.Length)
        {
            Debug.LogWarning($"Position array size mismatch: expected {targetPositions.Length}, got {positions.Length}");
            return;
        }

        // Set target positions
        for (int i = 0; i < positions.Length; i++)
        {
            targetPositions[i] = positions[i];
        }
    }

    public void SetVelocityCommand(float linearX, float linearY, float linearZ,
                                 float angularX, float angularY, float angularZ)
    {
        // Handle mobile base velocity commands
        // Implementation depends on robot base type
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = new Vector3(linearX, linearY, linearZ);
            rb.angularVelocity = new Vector3(angularX, angularY, angularZ);
        }
    }
}
```

## Safety and Error Handling

When integrating simulation with ROS2, safety and error handling are crucial for reliable operation:

1. **Connection Monitoring**: Continuously monitor connection status between systems
2. **Graceful Degradation**: Implement fallback behaviors when communication fails
3. **Command Validation**: Validate all incoming commands before execution
4. **Emergency Stop**: Implement emergency stop functionality for safety

## Conclusion

The integration of simulation environments with ROS2 enables powerful workflows for developing, testing, and validating humanoid robots. The ROS2-Gazebo and ROS2-Unity bridges provide robust communication channels that allow for sophisticated sensor simulation, control implementation, and real-time validation. Proper integration ensures that the digital twin accurately reflects the intended physical system behavior, enabling effective transfer of validated behaviors from simulation to reality.