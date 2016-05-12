cmake_minimum_required(VERSION 2.8.3)
project(drdo_planning)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  cv_bridge
  gazebo_msgs
  image_transport
  message_filters
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(drdo_afford src/drdo_afford.cpp)
target_link_libraries(drdo_afford ${catkin_LIBRARIES})
add_dependencies(drdo_afford ${catkin_EXPORTED_TARGETS})

add_executable(svm_test src/svm_test.cpp)
target_link_libraries(svm_test ${catkin_LIBRARIES})
add_dependencies(svm_test ${catkin_EXPORTED_TARGETS})

add_executable(learn src/learn.cpp)
target_link_libraries(learn ${catkin_LIBRARIES})
add_dependencies(learn ${catkin_EXPORTED_TARGETS})
