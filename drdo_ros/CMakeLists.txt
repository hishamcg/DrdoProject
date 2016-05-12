cmake_minimum_required(VERSION 2.8.3)
project(drdo_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(drdo_tf_broadcaster src/drdo_tf_broadcaster.cpp)
target_link_libraries(drdo_tf_broadcaster ${catkin_LIBRARIES})
add_dependencies(drdo_tf_broadcaster ${catkin_EXPORTED_TARGETS})

add_executable(drdo_odom_publisher src/drdo_odom_publisher.cpp)
target_link_libraries(drdo_odom_publisher ${catkin_LIBRARIES})
add_dependencies(drdo_odom_publisher ${catkin_EXPORTED_TARGETS})

add_executable(drdo_pid_controller src/drdo_pid_controller.cpp)
target_link_libraries(drdo_pid_controller ${catkin_LIBRARIES})
add_dependencies(drdo_pid_controller ${catkin_EXPORTED_TARGETS})
