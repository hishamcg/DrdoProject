rostopic pub -1 /pioneer/shoulder_pitch_joint_position_controller/command std_msgs/Float64 -- -0.0 &
rostopic pub -1 /pioneer/shoulder_roll_joint_position_controller/command std_msgs/Float64 -- 0.0 &
rostopic pub -1 /pioneer/shoulder_yaw_joint_position_controller/command std_msgs/Float64 -- 0.0 &
rostopic pub -1 /pioneer/elbow_pitch_joint_position_controller/command std_msgs/Float64 -- -0.0 &
rostopic pub -1 /pioneer/elbow_yaw_joint_position_controller/command std_msgs/Float64 -- -0.0 &
rostopic pub -1 /pioneer/wrist_pitch_joint_position_controller/command std_msgs/Float64 -- -0.0 &
rostopic pub -1 /pioneer/gripper_joint1_position_controller/command std_msgs/Float64 -- 0.0 &
rostopic pub -1 /pioneer/gripper_joint2_position_controller/command std_msgs/Float64 -- 0.0
