---
id: module-2-sensor-simulation
title: Chapter 5 - Sensor Simulation and Integration
sidebar_label: Chapter 5 - Sensor Simulation and Integration
---

# Chapter 5: Sensor Simulation and Integration

## Camera Sensor Simulation

Camera sensors are fundamental for humanoid robots, providing visual perception capabilities essential for navigation, object recognition, and environment understanding. Simulating camera sensors accurately in digital twin environments requires careful configuration of parameters to match physical sensors.

### Camera Simulation in Gazebo

In Gazebo, camera sensors are defined using the `<sensor>` tag with type "camera". Here's a complete example:

```xml
<!-- Camera sensor definition in URDF/SDF -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees in radians -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
      <update_rate>30.0</update_rate>
      <image_topic_name>/camera/image_raw</image_topic_name>
      <camera_info_topic_name>/camera/camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Simulation in Unity

In Unity, camera sensors are implemented using standard Unity cameras with ROS2 integration:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class CameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera unityCamera;
    public int width = 640;
    public int height = 480;
    public float updateRate = 30.0f;

    [Header("ROS2 Configuration")]
    public string imageTopic = "/camera/image_raw";
    public string cameraInfoTopic = "/camera/camera_info";

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSConnection ros;
    private float nextUpdate;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Set up camera if not assigned
        if (unityCamera == null)
            unityCamera = GetComponent<Camera>();

        // Create render texture for camera
        SetupCamera();
    }

    void SetupCamera()
    {
        // Create render texture
        renderTexture = new RenderTexture(width, height, 24);
        unityCamera.targetTexture = renderTexture;

        // Create texture2D for reading pixels
        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            CaptureAndPublishImage();
            nextUpdate = Time.time + (1.0f / updateRate);
        }
    }

    void CaptureAndPublishImage()
    {
        // Set active render texture
        RenderTexture.active = renderTexture;

        // Read pixels from render texture
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Convert to ROS image message
        var imageData = texture2D.GetRawTextureData<Color32>();

        // Create and publish ROS image message
        var rosImage = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.CompressedImageMsg
        {
            format = "jpeg",
            data = imageData
        };

        ros.Publish(imageTopic, rosImage);
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            RenderTexture.ReleaseTemporary(renderTexture);
        if (texture2D != null)
            Destroy(texture2D);
    }
}
```

## LIDAR Sensor Simulation

LIDAR sensors provide crucial 3D spatial information for navigation and mapping. Simulating LIDAR in digital twin environments requires accurate raycasting and point cloud generation.

### LIDAR Simulation in Gazebo

Gazebo provides several LIDAR sensor types, from simple 2D to complex 3D sensors:

```xml
<!-- 2D LIDAR sensor -->
<gazebo reference="lidar_link">
  <sensor name="lidar_2d" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
          <max_angle>3.14159</max_angle>    <!-- 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_2d_controller" filename="libgazebo_ros_laser.so">
      <frame_name>lidar_frame</frame_name>
      <topic_name>/scan</topic_name>
      <update_rate>10</update_rate>
    </plugin>
  </sensor>
</gazebo>

<!-- 3D LIDAR sensor (Velodyne-style) -->
<gazebo reference="velodyne_link">
  <sensor name="velodyne_3d" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1024</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle> <!-- -30 degrees -->
          <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
      <frame_name>velodyne_frame</frame_name>
      <topic_name>/velodyne_points</topic_name>
      <min_range>0.1</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.01</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR Simulation in Unity

In Unity, LIDAR simulation is implemented using raycasting:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class LIDARSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    public float minRange = 0.1f;
    public float maxRange = 30.0f;
    public int horizontalSamples = 720;
    public float horizontalFOV = 360f;
    public int verticalSamples = 1;
    public float verticalFOV = 0f;
    public float updateRate = 10.0f;

    [Header("ROS2 Configuration")]
    public string topicName = "/scan";

    private ROSConnection ros;
    private float nextUpdate;
    private RaycastHit[] raycastHits;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        raycastHits = new RaycastHit[horizontalSamples * verticalSamples];
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            PublishLIDARData();
            nextUpdate = Time.time + (1.0f / updateRate);
        }
    }

    void PublishLIDARData()
    {
        var ranges = new float[horizontalSamples * verticalSamples];

        // Calculate angles
        float horizontalStep = horizontalFOV / horizontalSamples;
        float verticalStep = verticalFOV / verticalSamples;

        for (int v = 0; v < verticalSamples; v++)
        {
            for (int h = 0; h < horizontalSamples; h++)
            {
                float hAngle = (h * horizontalStep - horizontalFOV / 2) * Mathf.Deg2Rad;
                float vAngle = (v * verticalStep - verticalFOV / 2) * Mathf.Deg2Rad;

                // Calculate ray direction
                Vector3 direction = CalculateLIDARDirection(hAngle, vAngle);

                // Perform raycast
                if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxRange))
                {
                    ranges[v * horizontalSamples + h] = hit.distance;
                }
                else
                {
                    ranges[v * horizontalSamples + h] = maxRange;
                }
            }
        }

        // Create and publish ROS LaserScan message
        var laserScan = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.LaserScanMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9f)
                },
                frame_id = transform.name
            },
            angle_min = -horizontalFOV * Mathf.Deg2Rad / 2,
            angle_max = horizontalFOV * Mathf.Deg2Rad / 2,
            angle_increment = (horizontalFOV * Mathf.Deg2Rad) / horizontalSamples,
            time_increment = 0,
            scan_time = 1.0f / updateRate,
            range_min = minRange,
            range_max = maxRange,
            ranges = ranges,
            intensities = new float[ranges.Length] // Intensity data (optional)
        };

        ros.Publish(topicName, laserScan);
    }

    Vector3 CalculateLIDARDirection(float hAngle, float vAngle)
    {
        // Convert spherical coordinates to Cartesian
        Vector3 direction = new Vector3(
            Mathf.Cos(vAngle) * Mathf.Cos(hAngle),
            Mathf.Cos(vAngle) * Mathf.Sin(hAngle),
            Mathf.Sin(vAngle)
        );

        // Apply transform rotation
        return transform.TransformDirection(direction);
    }
}
```

## IMU and Inertial Sensor Simulation

Inertial Measurement Units (IMUs) provide crucial orientation, acceleration, and angular velocity data for humanoid robot control and navigation.

### IMU Simulation in Gazebo

```xml
<!-- IMU sensor definition -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <frame_name>imu_link</frame_name>
      <topic_name>/imu/data</topic_name>
      <serviceName>/imu/service</serviceName>
      <gaussianNoise>0.0017</gaussianNoise>
      <updateRateHZ>100.0</updateRateHZ>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class IMUSensor : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float updateRate = 100.0f;
    public float noiseLevel = 0.001f;

    [Header("Noise Parameters")]
    public Vector3 angularVelocityNoise = new Vector3(2e-4f, 2e-4f, 2e-4f);
    public Vector3 linearAccelerationNoise = new Vector3(1.7e-2f, 1.7e-2f, 1.7e-2f);

    [Header("ROS2 Configuration")]
    public string topicName = "/imu/data";

    private ROSConnection ros;
    private float nextUpdate;
    private Rigidbody robotRigidbody;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Try to get rigidbody for more accurate IMU simulation
        robotRigidbody = GetComponentInParent<Rigidbody>();
        if (robotRigidbody == null)
            robotRigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            PublishIMUData();
            nextUpdate = Time.time + (1.0f / updateRate);
        }
    }

    void PublishIMUData()
    {
        // Get orientation (quaternion)
        Quaternion orientation = transform.rotation;

        // Get angular velocity (if rigidbody is available)
        Vector3 angularVelocity = Vector3.zero;
        if (robotRigidbody != null)
        {
            angularVelocity = transform.InverseTransformDirection(robotRigidbody.angularVelocity);
        }
        else
        {
            // Approximate angular velocity from rotation change
            angularVelocity = GetApproximateAngularVelocity();
        }

        // Get linear acceleration
        Vector3 linearAcceleration = GetLinearAcceleration();

        // Add noise to measurements
        angularVelocity += AddNoise(angularVelocityNoise);
        linearAcceleration += AddNoise(linearAccelerationNoise);

        // Create and publish ROS IMU message
        var imuMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.ImuMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9f)
                },
                frame_id = transform.name
            },
            orientation = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.QuaternionMsg
            {
                x = orientation.x,
                y = orientation.y,
                z = orientation.z,
                w = orientation.w
            },
            angular_velocity = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = angularVelocity.x,
                y = angularVelocity.y,
                z = angularVelocity.z
            },
            linear_acceleration = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = linearAcceleration.x,
                y = linearAcceleration.y,
                z = linearAcceleration.z
            }
        };

        ros.Publish(topicName, imuMsg);
    }

    Vector3 GetApproximateAngularVelocity()
    {
        // This is a simplified approximation
        // In a real implementation, you'd track rotation over time
        return new Vector3(
            Random.Range(-0.01f, 0.01f),
            Random.Range(-0.01f, 0.01f),
            Random.Range(-0.01f, 0.01f)
        );
    }

    Vector3 GetLinearAcceleration()
    {
        Vector3 acceleration = Vector3.zero;

        if (robotRigidbody != null)
        {
            // Calculate acceleration from velocity change
            // In practice, this would require storing previous velocity
            acceleration = transform.InverseTransformDirection(robotRigidbody.velocity);
        }
        else
        {
            // Use gravity and any applied forces
            acceleration = Physics.gravity;
        }

        return acceleration;
    }

    Vector3 AddNoise(Vector3 noiseLevels)
    {
        return new Vector3(
            Random.Range(-noiseLevels.x, noiseLevels.x),
            Random.Range(-noiseLevels.y, noiseLevels.y),
            Random.Range(-noiseLevels.z, noiseLevels.z)
        );
    }
}
```

## Force/Torque Sensor Simulation

Force/torque sensors are essential for humanoid robots to detect contact forces and enable safe interaction with the environment.

### Force/Torque Sensor Simulation in Gazebo

```xml
<!-- Force/Torque sensor in a joint -->
<gazebo reference="left_foot_joint">
  <sensor name="left_foot_ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>sensor</frame>
      <measure_direction>from_parent_link</measure_direction>
    </force_torque>
    <plugin name="ft_sensor_controller" filename="libgazebo_ros_ft_sensor.so">
      <frame_name>left_foot_link</frame_name>
      <topic_name>/left_foot/ft_sensor</topic_name>
      <update_rate>100.0</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Sensor Simulation in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class ForceTorqueSensor : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public float updateRate = 100.0f;
    public float noiseLevel = 0.1f;

    [Header("ROS2 Configuration")]
    public string topicName = "/ft_sensor";

    private ROSConnection ros;
    private float nextUpdate;
    private Collider sensorCollider;
    private ContactPoint[] contacts = new ContactPoint[10];
    private int contactCount = 0;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        sensorCollider = GetComponent<Collider>();
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            PublishForceTorqueData();
            nextUpdate = Time.time + (1.0f / updateRate);
        }
    }

    void PublishForceTorqueData()
    {
        // Calculate forces from contacts
        Vector3 totalForce = Vector3.zero;
        Vector3 totalTorque = Vector3.zero;

        // Get contact information (this would need to be updated based on actual contacts)
        totalForce = CalculateContactForce();
        totalTorque = CalculateContactTorque(totalForce);

        // Add noise to measurements
        totalForce += new Vector3(
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel)
        );

        totalTorque += new Vector3(
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel)
        );

        // Create and publish ROS Wrench message
        var wrenchMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.WrenchMsg
        {
            force = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = totalForce.x,
                y = totalForce.y,
                z = totalForce.z
            },
            torque = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = totalTorque.x,
                y = totalTorque.y,
                z = totalTorque.z
            }
        };

        ros.Publish(topicName, wrenchMsg);
    }

    Vector3 CalculateContactForce()
    {
        // In a real implementation, this would calculate force from actual contacts
        // For simulation, we'll use a simplified approach
        var colliders = Physics.OverlapBox(transform.position, sensorCollider.bounds.extents);

        Vector3 totalForce = Vector3.zero;
        foreach (var collider in colliders)
        {
            if (collider.gameObject != gameObject)
            {
                // Calculate contact force based on collision
                totalForce += Physics.gravity * 0.1f; // Simplified
            }
        }

        return totalForce;
    }

    Vector3 CalculateContactTorque(Vector3 force)
    {
        // Calculate torque as cross product of position and force
        return Vector3.Cross(transform.position, force);
    }
}
```

## Multi-Sensor Integration

Integrating multiple sensors requires careful synchronization and data fusion to provide coherent perception capabilities.

### Sensor Synchronization in Unity

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MultiSensorFusion : MonoBehaviour
{
    [Header("Sensor Components")]
    public List<SensorBase> sensors = new List<SensorBase>();

    [Header("Synchronization")]
    public float synchronizationWindow = 0.01f; // 10ms window

    private Dictionary<string, SensorData> latestSensorData = new Dictionary<string, SensorData>();
    private float lastSyncTime;

    void Start()
    {
        InitializeSensors();
    }

    void InitializeSensors()
    {
        // Find all sensor components attached to this robot
        var sensorComponents = GetComponentsInChildren<SensorBase>();
        foreach (var sensor in sensorComponents)
        {
            sensors.Add(sensor);
            latestSensorData[sensor.GetSensorId()] = new SensorData();
        }
    }

    void Update()
    {
        CheckSynchronization();
    }

    void CheckSynchronization()
    {
        // Check if all sensors have updated within the synchronization window
        bool allUpdated = true;
        float currentTime = Time.time;

        foreach (var sensor in sensors)
        {
            if (currentTime - sensor.LastUpdateTime > synchronizationWindow)
            {
                allUpdated = false;
                break;
            }
        }

        if (allUpdated && currentTime - lastSyncTime > synchronizationWindow)
        {
            // All sensors are synchronized, perform fusion
            PerformSensorFusion();
            lastSyncTime = currentTime;
        }
    }

    void PerformSensorFusion()
    {
        // Example: Combine IMU and camera data for better pose estimation
        var imuData = GetLatestSensorData("IMU");
        var cameraData = GetLatestSensorData("Camera");

        if (imuData != null && cameraData != null)
        {
            // Perform sensor fusion algorithm
            var fusedPose = FuseIMUCameraData(imuData, cameraData);

            // Publish fused data
            PublishFusedData(fusedPose);
        }
    }

    SensorData GetLatestSensorData(string sensorType)
    {
        foreach (var pair in latestSensorData)
        {
            if (pair.Key.Contains(sensorType))
            {
                return pair.Value;
            }
        }
        return null;
    }

    Pose FuseIMUCameraData(SensorData imuData, SensorData cameraData)
    {
        // Simplified sensor fusion algorithm
        // In practice, this would use more sophisticated methods like Kalman filtering
        return new Pose(Vector3.zero, Quaternion.identity);
    }

    void PublishFusedData(Pose fusedPose)
    {
        // Publish the fused sensor data to ROS2 or other systems
        Debug.Log($"Publishing fused pose: {fusedPose}");
    }
}

public abstract class SensorBase : MonoBehaviour
{
    public string sensorId;
    public float LastUpdateTime { get; protected set; }

    public string GetSensorId()
    {
        return sensorId;
    }

    protected virtual void Start()
    {
        LastUpdateTime = Time.time;
    }
}

[System.Serializable]
public class SensorData
{
    public float timestamp;
    public object data;
    public string sensorType;
}
```

## Performance Optimization for Sensor Simulation

Simulating multiple sensors can be computationally expensive. Here are optimization techniques:

### Adaptive Update Rates

```csharp
using UnityEngine;

public class AdaptiveSensorUpdater : MonoBehaviour
{
    public enum SensorPriority
    {
        Critical,     // High update rate (100Hz+)
        Standard,     // Medium update rate (30Hz)
        LowPriority   // Low update rate (10Hz)
    }

    [System.Serializable]
    public class SensorConfiguration
    {
        public SensorBase sensor;
        public SensorPriority priority;
        public float baseUpdateRate;
        public float minUpdateRate;
        public float maxUpdateRate;
    }

    public List<SensorConfiguration> sensorConfigs = new List<SensorConfiguration>();

    void Start()
    {
        SetupAdaptiveUpdateRates();
    }

    void SetupAdaptiveUpdateRates()
    {
        foreach (var config in sensorConfigs)
        {
            // Set initial update rates based on priority
            switch (config.priority)
            {
                case SensorPriority.Critical:
                    config.baseUpdateRate = 100f;
                    config.minUpdateRate = 50f;
                    config.maxUpdateRate = 200f;
                    break;
                case SensorPriority.Standard:
                    config.baseUpdateRate = 30f;
                    config.minUpdateRate = 10f;
                    config.maxUpdateRate = 60f;
                    break;
                case SensorPriority.LowPriority:
                    config.baseUpdateRate = 10f;
                    config.minUpdateRate = 5f;
                    config.maxUpdateRate = 30f;
                    break;
            }
        }
    }

    void Update()
    {
        // Adjust update rates based on system performance
        AdjustUpdateRates();
    }

    void AdjustUpdateRates()
    {
        float performanceFactor = CalculatePerformanceFactor();

        foreach (var config in sensorConfigs)
        {
            float adjustedRate = config.baseUpdateRate * performanceFactor;
            adjustedRate = Mathf.Clamp(adjustedRate, config.minUpdateRate, config.maxUpdateRate);

            // Apply rate to sensor (implementation depends on sensor type)
            if (config.sensor is CameraSensor camSensor)
            {
                camSensor.updateRate = adjustedRate;
            }
            else if (config.sensor is LIDARSensor lidarSensor)
            {
                lidarSensor.updateRate = adjustedRate;
            }
        }
    }

    float CalculatePerformanceFactor()
    {
        // Calculate performance factor based on frame rate
        float targetFrameRate = 60f;
        float currentFrameRate = 1f / Time.deltaTime;
        return Mathf.Clamp01(currentFrameRate / targetFrameRate);
    }
}
```

## Conclusion

Sensor simulation is a critical component of digital twin systems for humanoid robots, providing the perception capabilities necessary for autonomous operation. Accurate simulation of cameras, LIDAR, IMU, and force/torque sensors ensures that control algorithms and perception systems can be thoroughly tested in virtual environments before deployment on physical robots. Proper configuration of sensor parameters, noise models, and update rates is essential for achieving realistic simulation that closely matches physical sensor behavior.