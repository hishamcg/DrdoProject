cmake_minimum_required(VERSION 2.8.3)
project(pioneer_cyton_gazebo)

find_package(gazebo REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  angles
  control_toolbox
  gazebo_ros
  robot_controllers
  robot_controllers_interface
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)


link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

catkin_package(
   CATKIN_DEPENDS
    roscpp
    rospy
    control_toolbox
    gazebo_ros
    robot_controllers
    robot_controllers_interface
  DEPENDS
    gazebo
  LIBRARIES
    pioneer_cyton_gazebo_plugin
  INCLUDE_DIRS
    include
)

add_library(pioneer_cyton_gazebo_plugin src/plugin.cpp)
target_link_libraries(pioneer_cyton_gazebo_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

install(
  TARGETS pioneer_cyton_gazebo_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}  
)

install(
  PROGRAMS
    scripts/prepare_simulated_robot.py
    scripts/demo.py
    scripts/demo_move_group.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
