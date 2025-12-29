---
id: module-1-nodes-lifecycle
title: Chapter 2 - Nodes and Lifecycle Management
sidebar_label: Chapter 2 - Nodes and Lifecycle Management
---

# Chapter 2: Nodes and Lifecycle Management

## Node Creation in C++

In ROS2, a node serves as the fundamental building block of any robotic application. A node is an executable process that communicates with other nodes through topics, services, actions, and parameters. Creating a node in C++ involves inheriting from the `rclcpp::Node` class and implementing the desired functionality.

### Basic Node Structure

A basic ROS2 node in C++ follows this structure:

```cpp
#include <rclcpp/rclcpp.hpp>

class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("node_name")
    {
        // Initialize node components here
    }

private:
    // Declare member variables for publishers, subscribers, etc.
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Node Parameters and Configuration

ROS2 nodes can declare and use parameters to make them configurable without recompilation. Parameters can be set at runtime through launch files, command-line arguments, or programmatically.

```cpp
// Declare a parameter with a default value
this->declare_parameter<std::string>("robot_name", "my_robot");
this->declare_parameter<double>("loop_rate", 10.0);

// Get parameter value
std::string robot_name = this->get_parameter("robot_name").as_string();
double loop_rate = this->get_parameter("loop_rate").as_double();
```

### Node Composition

ROS2 supports node composition, allowing multiple nodes to run within the same process. This approach reduces communication overhead and can improve performance for tightly coupled components.

## Node Creation in Python

Creating nodes in Python follows a similar pattern but with Python-specific syntax and conventions.

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):

    def __init__(self):
        super().__init__('node_name')
        # Initialize node components here

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Async/Await Patterns in Python Nodes

Python nodes can leverage async/await patterns for more efficient handling of concurrent operations:

```python
import rclpy
from rclpy.node import Node
import asyncio

class AsyncNode(Node):

    def __init__(self):
        super().__init__('async_node')

    async def async_operation(self):
        # Perform asynchronous operation
        await asyncio.sleep(1.0)
        self.get_logger().info('Async operation completed')

def main(args=None):
    rclpy.init(args=args)
    node = AsyncNode()

    # Run async operations
    loop = asyncio.get_event_loop()
    loop.run_until_complete(node.async_operation())

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Node Lifecycle Management

ROS2 introduces a sophisticated lifecycle management system that provides better control over node states and transitions. The lifecycle node concept addresses challenges in complex robotic systems where components need to be initialized, activated, deactivated, and cleaned up in a coordinated manner.

### Lifecycle Node States

The ROS2 lifecycle system defines several states that a node can occupy:

- **Unconfigured**: Initial state after node creation
- **Inactive**: Node is configured but not active
- **Active**: Node is fully operational
- **Finalized**: Node is shutting down

### Lifecycle Node Implementation in C++

```cpp
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

class LifecycleNodeExample : public rclcpp_lifecycle::LifecycleNode
{
public:
    LifecycleNodeExample() : rclcpp_lifecycle::LifecycleNode("lifecycle_node")
    {
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Configuring node");
        // Initialize resources but don't start processing
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Activating node");
        // Start processing, activate publishers/subscribers
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Deactivating node");
        // Stop processing, deactivate publishers/subscribers
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Cleaning up node");
        // Release resources
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Shutting down node");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};
```

### Lifecycle Node Implementation in Python

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from std_msgs.msg import String

class LifecycleNodeExample(LifecycleNode):

    def __init__(self):
        super().__init__('lifecycle_node')
        self.pub = None

    def on_configure(self, state):
        self.get_logger().info('Configuring node')
        # Initialize resources but don't start processing
        self.pub = self.create_publisher(String, 'topic', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node')
        # Start processing, activate publishers/subscribers
        self.pub.on_activate()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node')
        # Stop processing, deactivate publishers/subscribers
        self.pub.on_deactivate()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up node')
        # Release resources
        self.destroy_publisher(self.pub)
        self.pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info('Shutting down node')
        return TransitionCallbackReturn.SUCCESS
```

## Parameters and Configuration

### Parameter Server Functionality

The ROS2 parameter server provides a centralized mechanism for managing node parameters. Parameters can be:

- Declared at runtime with default values
- Modified dynamically during execution
- Loaded from YAML configuration files
- Remapped using namespaces

### Dynamic Parameter Updates

Nodes can be configured to respond to parameter changes at runtime:

```cpp
// C++ example of parameter callback
auto param_change_callback = [this](const std::vector<rclcpp::Parameter> &parameters) {
    for (const auto &parameter : parameters) {
        if (parameter.get_name() == "loop_rate") {
            loop_rate_ = parameter.as_double();
        }
    }
    return rclcpp_interfaces::msg::SetParametersResult();
};

this->set_on_parameters_set_callback(param_change_callback);
```

```python
# Python example of parameter callback
def parameter_callback(self, parameters):
    for param in parameters:
        if param.name == 'loop_rate':
            self.loop_rate = param.value
    return SetParametersResult(successful=True)

self.set_parameters_callback(parameter_callback)
```

## Best Practices for Node Development

### Error Handling

Implement proper error handling in nodes to ensure robust operation:

```cpp
// C++ error handling example
try {
    auto result = client_->async_send_request(request);
    if (result.get()) {
        RCLCPP_INFO(this->get_logger(), "Service call successful");
    } else {
        RCLCPP_ERROR(this->get_logger(), "Service call failed");
    }
} catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Exception during service call: %s", e.what());
}
```

### Resource Management

Properly manage resources to prevent memory leaks and ensure clean shutdown:

- Always destroy publishers, subscribers, and clients in the node destructor
- Use smart pointers where appropriate
- Implement proper cleanup in lifecycle nodes

## Conclusion

Node creation and lifecycle management form the foundation of ROS2 applications. Understanding these concepts is crucial for developing robust, maintainable robotic systems. The lifecycle node pattern provides enhanced control over node states, making it particularly valuable for complex robotic applications where coordinated initialization and shutdown procedures are essential.

The choice between basic nodes and lifecycle nodes depends on the complexity of your application and the level of control required over the node's state transitions. For simple applications, basic nodes may suffice, but for production systems, lifecycle nodes provide better management capabilities.