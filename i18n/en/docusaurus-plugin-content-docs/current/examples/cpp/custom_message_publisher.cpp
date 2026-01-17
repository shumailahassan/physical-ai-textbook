#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

class CustomMessagePublisher : public rclcpp::Node
{
public:
    CustomMessagePublisher()
    : Node("custom_message_publisher")
    {
        // Create publisher with custom QoS settings
        rclcpp::QoS qos_profile(10);
        qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

        publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "humanoid_joints", qos_profile);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&CustomMessagePublisher::publish_joint_states, this));
    }

private:
    void publish_joint_states()
    {
        auto msg = sensor_msgs::msg::JointState();
        msg.name = {"hip_left", "knee_left", "ankle_left", "hip_right", "knee_right", "ankle_right"};
        msg.position = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
        msg.velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        msg.effort = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        publisher_->publish(msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CustomMessagePublisher>());
    rclcpp::shutdown();
    return 0;
}