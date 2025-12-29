#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

class QoSSubscriber : public rclcpp::Node
{
public:
    QoSSubscriber()
    : Node("qos_subscriber")
    {
        // Create subscriber with matching QoS settings
        rclcpp::QoS qos_profile(10);
        qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "humanoid_joints", qos_profile,
            std::bind(&QoSSubscriber::joint_state_callback, this, std::placeholders::_1));
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(),
            "Received joint states - Joint 1: %.2f, Joint 2: %.2f",
            msg->position[0], msg->position[1]);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QoSSubscriber>());
    rclcpp::shutdown();
    return 0;
}