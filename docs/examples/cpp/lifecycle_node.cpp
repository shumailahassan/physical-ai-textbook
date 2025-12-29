#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/string.hpp>

class LifecycleNodeExample : public rclcpp_lifecycle::LifecycleNode
{
public:
    LifecycleNodeExample() : rclcpp_lifecycle::LifecycleNode("lifecycle_node")
    {
        RCLCPP_INFO(get_logger(), "Lifecycle node created");
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Configuring node");
        // Initialize resources but don't start processing
        pub_ = this->create_publisher<std_msgs::msg::String>("lifecycle_topic", 10);
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Activating node");
        // Start processing, activate publishers/subscribers
        pub_->on_activate();
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Deactivating node");
        // Stop processing, deactivate publishers/subscribers
        pub_->on_deactivate();
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Cleaning up node");
        // Release resources
        pub_.reset();
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Shutting down node");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

private:
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LifecycleNodeExample>();

    // Manually trigger lifecycle transitions for demonstration
    node->configure();
    node->activate();

    // Create a timer to publish messages while active
    auto timer = node->create_wall_timer(
        std::chrono::seconds(1),
        [&node]() {
            if (node->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
                auto msg = std_msgs::msg::String();
                msg.data = "Lifecycle message";
                node->pub_->publish(msg);
            }
        }
    );

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}