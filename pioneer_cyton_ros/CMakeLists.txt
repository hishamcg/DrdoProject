cmake_minimum_required(VERSION 2.8.3)
project(pioneer_cyton_ros)

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

add_executable(pioneer_cyton_tf_broadcaster src/pioneer_cyton_tf_broadcaster.cpp)
target_link_libraries(pioneer_cyton_tf_broadcaster ${catkin_LIBRARIES})
add_dependencies(pioneer_cyton_tf_broadcaster ${catkin_EXPORTED_TARGETS})

add_executable(pioneer_cyton_odom_publisher src/pioneer_cyton_odom_publisher.cpp)
target_link_libraries(pioneer_cyton_odom_publisher ${catkin_LIBRARIES})
add_dependencies(pioneer_cyton_odom_publisher ${catkin_EXPORTED_TARGETS})

add_executable(pioneer_cyton_pid_controller src/pioneer_cyton_pid_controller.cpp)
target_link_libraries(pioneer_cyton_pid_controller ${catkin_LIBRARIES})
add_dependencies(pioneer_cyton_pid_controller ${catkin_EXPORTED_TARGETS})

##add_executable(pioneer_cyton_pick_place src/pioneer_cyton_pick_place.cpp)
##target_link_libraries(pioneer_cyton_pick_place ${catkin_LIBRARIES} ${Boost_LIBRARIES})
