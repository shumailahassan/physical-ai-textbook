---
# Translation needed
# اس صفحے کا ترجمہ درکار ہے
---

**This page is currently only available in English. Please contribute a translation.**
**یہ صفحہ فی الحال صرف انگریزی میں دستیاب ہے۔ براہ کرم ایک ترجمہ شامل کریں۔**

---
id: module-2-robot-modeling-physics
title: Chapter 4 - Robot Modeling and Physics
sidebar_label: Chapter 4 - Robot Modeling and Physics
---

# Chapter 4: Robot Modeling and Physics

## Robot Modeling in Gazebo

Creating accurate robot models for Gazebo simulation requires careful attention to both visual representation and physical properties. The Unified Robot Description Format (URDF) serves as the standard for defining robot models in ROS-based systems, which are then used by Gazebo for physics simulation.

### URDF Format for Robot Description

URDF (Unified Robot Description Format) is an XML-based format that describes robot models. A complete humanoid robot model in URDF includes links (rigid bodies) and joints (constraints between links).

Here's a detailed example of a humanoid robot's URDF structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link definition -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso link -->
  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0"
               iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Hip joint and leg links -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="-0.05 -0.1 0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0"
               iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Additional joints and links for complete humanoid model -->
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.5" effort="100" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0"
               iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
  </link>

  <!-- Foot joint and link -->
  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0"
               iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_thigh">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="left_shin">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_foot">
    <material>Gazebo/Yellow</material>
  </gazebo>

  <!-- Gazebo plugins for control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

### Link and Joint Definitions

In the URDF format:

- **Links** represent rigid bodies with mass, inertia, and geometry
- **Joints** define the connection between links with specific degrees of freedom
- **Visual** elements define how the robot appears in simulation
- **Collision** elements define physical interaction boundaries

### Collision and Visual Geometries

Proper configuration of collision and visual geometries is crucial for realistic simulation:

- **Visual geometries** determine how the robot appears (for rendering)
- **Collision geometries** determine physical interactions (for physics simulation)
- Collision geometries should be simplified compared to visual geometries for performance

## Robot Modeling in Unity

Unity provides different approaches for robot modeling, including direct import of URDF files and manual creation of robot components.

### Importing URDF Models to Unity

The Unity URDF Importer package enables direct import of ROS robot models:

```csharp
using UnityEngine;
using Unity.Robotics.URDFImporter;

public class RobotModelLoader : MonoBehaviour
{
    [Header("URDF Configuration")]
    public string urdfPath;
    public string robotName;

    [Header("Import Settings")]
    public bool createArticulations = true;
    public bool useCollision = true;
    public bool useInertia = true;

    void Start()
    {
        LoadRobotModel();
    }

    public void LoadRobotModel()
    {
        if (!string.IsNullOrEmpty(urdfPath))
        {
            // Load URDF file and create robot model
            var robot = URDFRobotExtensions.CreateRobot(urdfPath);

            if (robot != null)
            {
                robot.transform.SetParent(transform);
                ConfigureRobotPhysics(robot);
            }
        }
    }

    void ConfigureRobotPhysics(GameObject robot)
    {
        // Configure rigidbodies for each link
        var links = robot.GetComponentsInChildren<LinkComponent>();
        foreach (var link in links)
        {
            ConfigureLinkPhysics(link);
        }

        // Configure joints between links
        var joints = robot.GetComponentsInChildren<JointComponent>();
        foreach (var joint in joints)
        {
            ConfigureJointConstraints(joint);
        }
    }

    void ConfigureLinkPhysics(LinkComponent link)
    {
        var rb = link.GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = link.gameObject.AddComponent<Rigidbody>();
        }

        // Set physical properties from URDF
        rb.mass = link.urdfMass;
        rb.centerOfMass = link.urdfCenterOfMass;
        rb.inertiaTensor = link.urdfInertiaTensor;
    }

    void ConfigureJointConstraints(JointComponent joint)
    {
        var jointComponent = joint.GetComponent<UnityEngine.Joint>();
        if (jointComponent != null)
        {
            // Configure joint based on URDF joint type
            switch (joint.urdfJointType)
            {
                case JointType.Revolute:
                case JointType.Continuous:
                    var hingeJoint = jointComponent as HingeJoint;
                    if (hingeJoint != null)
                    {
                        hingeJoint.limits = new JointLimits
                        {
                            min = joint.urdfLowerLimit,
                            max = joint.urdfUpperLimit,
                            bounciness = 0.1f
                        };
                        hingeJoint.useLimits = true;
                        hingeJoint.useSpring = true;
                        hingeJoint.spring = new JointSpring
                        {
                            spring = joint.urdfEffortLimit * 0.1f,
                            damper = joint.urdfVelocityLimit * 0.1f,
                            targetPosition = 0
                        };
                    }
                    break;

                case JointType.Prismatic:
                    var configurableJoint = jointComponent as ConfigurableJoint;
                    if (configurableJoint != null)
                    {
                        configurableJoint.linearLimit = new SoftJointLimit
                        {
                            limit = joint.urdfUpperLimit
                        };
                    }
                    break;
            }
        }
    }
}
```

### Manual Robot Creation in Unity

For custom robot models or when not using URDF:

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class HumanoidLink : MonoBehaviour
{
    [Header("Physical Properties")]
    public float mass = 1.0f;
    public Vector3 centerOfMass = Vector3.zero;
    public Vector3 inertiaTensor = Vector3.one;

    [Header("Joint Configuration")]
    public JointType jointType = JointType.Revolute;
    public Vector3 jointAxis = Vector3.right;

    [Header("Joint Limits")]
    [Range(-180f, 180f)] public float lowerLimit = -90f;
    [Range(-180f, 180f)] public float upperLimit = 90f;

    [Header("Actuation")]
    public float maxEffort = 100f;
    public float maxVelocity = 2f;

    private Rigidbody rb;
    private Joint joint;

    void Start()
    {
        InitializePhysics();
        ConfigureJoint();
    }

    void InitializePhysics()
    {
        rb = GetComponent<Rigidbody>();
        rb.mass = mass;
        rb.centerOfMass = centerOfMass;
        rb.inertiaTensor = inertiaTensor;

        // Add collision if needed
        if (GetComponent<Collider>() == null)
        {
            // Add appropriate collider based on link shape
            var meshFilter = GetComponent<MeshFilter>();
            if (meshFilter != null && meshFilter.sharedMesh != null)
            {
                gameObject.AddComponent<MeshCollider>();
            }
            else
            {
                // Default to box collider
                var boxCollider = gameObject.AddComponent<BoxCollider>();
                boxCollider.center = centerOfMass;
            }
        }
    }

    void ConfigureJoint()
    {
        // Configure joint based on type
        switch (jointType)
        {
            case JointType.Revolute:
                ConfigureHingeJoint();
                break;
            case JointType.Prismatic:
                ConfigurePrismaticJoint();
                break;
            case JointType.Fixed:
                ConfigureFixedJoint();
                break;
        }
    }

    void ConfigureHingeJoint()
    {
        var hingeJoint = gameObject.AddComponent<HingeJoint>();
        hingeJoint.axis = jointAxis;
        hingeJoint.limits = new JointLimits
        {
            min = lowerLimit,
            max = upperLimit,
            bounciness = 0.1f
        };
        hingeJoint.useLimits = true;
        hingeJoint.motor = new JointMotor
        {
            targetVelocity = 0,
            force = maxEffort,
            freeSpin = false
        };
        hingeJoint.useMotor = true;

        joint = hingeJoint;
    }

    void ConfigurePrismaticJoint()
    {
        var configurableJoint = gameObject.AddComponent<ConfigurableJoint>();
        configurableJoint.axis = jointAxis;
        configurableJoint.secondaryAxis = Vector3.Cross(jointAxis, Vector3.up);

        configurableJoint.linearLimit = new SoftJointLimit
        {
            limit = Mathf.Max(Mathf.Abs(lowerLimit), Mathf.Abs(upperLimit))
        };

        joint = configurableJoint;
    }

    void ConfigureFixedJoint()
    {
        var fixedJoint = gameObject.AddComponent<FixedJoint>();
        joint = fixedJoint;
    }

    public void ApplyJointTorque(float torque)
    {
        if (joint is HingeJoint hingeJoint)
        {
            var motor = hingeJoint.motor;
            motor.targetVelocity = Mathf.Clamp(torque / maxEffort * maxVelocity, -maxVelocity, maxVelocity);
            motor.force = Mathf.Abs(torque);
            hingeJoint.motor = motor;
            hingeJoint.useMotor = true;
        }
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

## Physics Configuration and Tuning

Proper physics configuration is essential for realistic simulation behavior that matches physical robot characteristics.

### Physics Parameters for Realistic Simulation

The physics parameters should match the physical robot's characteristics:

- **Mass**: Accurate mass values for each link
- **Inertia**: Proper moment of inertia tensors
- **Damping**: Appropriate damping coefficients for joints
- **Friction**: Realistic friction values for contact surfaces

### Damping and Friction Configuration

Damping and friction parameters affect robot behavior significantly:

```csharp
using UnityEngine;

public class PhysicsTuner : MonoBehaviour
{
    [Header("Joint Damping Configuration")]
    public float globalDamping = 0.1f;
    public float globalFriction = 0.0f;

    [Header("Contact Properties")]
    public float contactStiffness = 100000f;
    public float contactDamping = 1000f;

    public void ConfigurePhysicsParameters()
    {
        // Configure global physics settings
        Physics.defaultSolverIterations = 10;
        Physics.defaultSolverVelocityIterations = 20;
        Physics.sleepThreshold = 0.005f;
        Physics.defaultContactOffset = 0.01f;

        // Apply to all rigidbodies in robot
        var rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (var rb in rigidbodies)
        {
            ConfigureRigidbody(rb);
        }

        // Configure all joints in robot
        var joints = GetComponentsInChildren<Joint>();
        foreach (var joint in joints)
        {
            ConfigureJoint(joint);
        }
    }

    void ConfigureRigidbody(Rigidbody rb)
    {
        // Set damping properties
        rb.drag = globalDamping;
        rb.angularDrag = globalDamping * 0.5f;
    }

    void ConfigureJoint(Joint joint)
    {
        // Configure joint-specific damping
        if (joint is HingeJoint hinge)
        {
            var spring = hinge.spring;
            spring.spring = globalDamping * 100f;
            spring.damper = globalDamping * 10f;
            hinge.spring = spring;
        }
        else if (joint is ConfigurableJoint configurable)
        {
            configurable.angularXMotion = ConfigurableJointMotion.Limited;
            configurable.angularYMotion = ConfigurableJointMotion.Limited;
            configurable.angularZMotion = ConfigurableJointMotion.Limited;

            // Set angular limits with damping
            var angularLimit = configurable.angularXLimit;
            angularLimit.limit = 45f; // degrees
            configurable.angularXLimit = angularLimit;
        }
    }

    public void MatchPhysicalRobot(HumanoidRobotSpec physicalSpec)
    {
        // Apply physical robot specifications to simulation
        var links = GetComponentsInChildren<HumanoidLink>();

        foreach (var link in links)
        {
            var spec = physicalSpec.GetLinkSpec(link.name);
            if (spec != null)
            {
                link.mass = spec.mass;
                link.centerOfMass = spec.centerOfMass;
                link.inertiaTensor = spec.inertiaTensor;

                // Update rigidbody with new values
                var rb = link.GetComponent<Rigidbody>();
                if (rb != null)
                {
                    rb.mass = spec.mass;
                    rb.centerOfMass = spec.centerOfMass;
                    rb.inertiaTensor = spec.inertiaTensor;
                }
            }
        }
    }
}

[System.Serializable]
public class HumanoidRobotSpec
{
    public LinkSpecification[] links;
    public JointSpecification[] joints;

    public LinkSpecification GetLinkSpec(string linkName)
    {
        foreach (var link in links)
        {
            if (link.name == linkName)
                return link;
        }
        return null;
    }
}

[System.Serializable]
public class LinkSpecification
{
    public string name;
    public float mass;
    public Vector3 centerOfMass;
    public Vector3 inertiaTensor;
    public float friction;
    public float bounciness;
}

[System.Serializable]
public class JointSpecification
{
    public string name;
    public JointType jointType;
    public float lowerLimit;
    public float upperLimit;
    public float effortLimit;
    public float velocityLimit;
    public float damping;
    public float friction;
}
```

## Collision Geometry and Visual Meshes

Separating collision geometry from visual meshes is important for performance and accuracy.

### Optimization Techniques

For collision detection optimization:

- Use simplified geometries (boxes, spheres, capsules) for collision
- Use detailed meshes only for visualization
- Implement level-of-detail (LOD) systems for complex models
- Use compound colliders for complex shapes

### Performance Considerations

```csharp
using UnityEngine;

public class CollisionOptimizer : MonoBehaviour
{
    [Header("Optimization Settings")]
    public bool optimizeColliders = true;
    public bool useCompoundColliders = true;
    public float simplificationFactor = 0.1f;

    void Start()
    {
        if (optimizeColliders)
        {
            OptimizeRobotColliders();
        }
    }

    void OptimizeRobotColliders()
    {
        var links = GetComponentsInChildren<HumanoidLink>();

        foreach (var link in links)
        {
            OptimizeLinkCollider(link);
        }
    }

    void OptimizeLinkCollider(HumanoidLink link)
    {
        var originalCollider = link.GetComponent<Collider>();
        if (originalCollider == null) return;

        // For complex meshes, replace with simpler collision shapes
        var meshCollider = originalCollider as MeshCollider;
        if (meshCollider != null && meshCollider.sharedMesh != null)
        {
            // Try to replace with primitive colliders if possible
            var bounds = meshCollider.sharedMesh.bounds;
            var size = bounds.size;

            // Determine best primitive based on shape
            if (IsApproximatelyCylindrical(size))
            {
                ReplaceWithCapsuleCollider(link.gameObject, bounds);
            }
            else if (IsApproximatelyBoxShaped(size))
            {
                ReplaceWithBoxCollider(link.gameObject, bounds);
            }
            else
            {
                // Keep mesh collider but simplify if possible
                SimplifyMeshCollider(meshCollider);
            }
        }
    }

    bool IsApproximatelyCylindrical(Vector3 size)
    {
        // Check if two dimensions are similar and one is larger
        var dimensions = new float[] { size.x, size.y, size.z };
        System.Array.Sort(dimensions);
        return (dimensions[1] / dimensions[0]) < 1.5f && (dimensions[2] / dimensions[1]) > 2f;
    }

    bool IsApproximatelyBoxShaped(Vector3 size)
    {
        // Check if all dimensions are within reasonable bounds
        var ratioXY = size.x / size.y;
        var ratioXZ = size.x / size.z;
        var ratioYZ = size.y / size.z;

        return Mathf.Abs(ratioXY - 1f) < 0.5f &&
               Mathf.Abs(ratioXZ - 1f) < 0.5f &&
               Mathf.Abs(ratioYZ - 1f) < 0.5f;
    }

    void ReplaceWithCapsuleCollider(GameObject link, Bounds bounds)
    {
        var capsuleCollider = link.AddComponent<CapsuleCollider>();

        // Calculate capsule dimensions
        var maxSize = Mathf.Max(bounds.size.x, bounds.size.z);
        var height = bounds.size.y;

        capsuleCollider.radius = maxSize / 2f;
        capsuleCollider.height = height;
        capsuleCollider.direction = 1; // Y-axis
        capsuleCollider.center = bounds.center - (Vector3)link.transform.position;

        // Remove original collider
        var oldCollider = link.GetComponent<Collider>();
        if (oldCollider != null)
            DestroyImmediate(oldCollider);
    }

    void ReplaceWithBoxCollider(GameObject link, Bounds bounds)
    {
        var boxCollider = link.AddComponent<BoxCollider>();
        boxCollider.size = bounds.size;
        boxCollider.center = bounds.center - (Vector3)link.transform.position;

        // Remove original collider
        var oldCollider = link.GetComponent<Collider>();
        if (oldCollider != null)
            DestroyImmediate(oldCollider);
    }

    void SimplifyMeshCollider(MeshCollider meshCollider)
    {
        // In a real implementation, this would simplify the mesh
        // For now, we'll just keep the original but mark it as optimized
        meshCollider.convex = true; // Use convex hull for better performance
    }
}
```

## Conclusion

Proper robot modeling and physics configuration are fundamental to creating accurate digital twins for humanoid robots. Whether using Gazebo with URDF or Unity with custom components, the key is to ensure that the virtual model accurately reflects the physical robot's characteristics, including mass distribution, joint constraints, and physical interactions. This accuracy is essential for the digital twin to provide meaningful insights and enable effective testing and validation of robot behaviors before deployment on the physical system.