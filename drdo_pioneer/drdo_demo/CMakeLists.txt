cmake_minimum_required(VERSION 2.8.3)
project(drdo_demo)

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

add_executable(drdo_demo src/drdo_demo.cpp)
target_link_libraries(drdo_demo ${catkin_LIBRARIES})
add_dependencies(drdo_demo ${catkin_EXPORTED_TARGETS})

add_executable(cluster_demo src/cluster_demo.cpp)
target_link_libraries(cluster_demo ${catkin_LIBRARIES})
add_dependencies(cluster_demo ${catkin_EXPORTED_TARGETS})

add_executable(svm_demo src/svm_demo.cpp)
target_link_libraries(svm_demo ${catkin_LIBRARIES})
add_dependencies(svm_demo ${catkin_EXPORTED_TARGETS})

add_executable(kmean_test src/kmean_test.cpp)
target_link_libraries(kmean_test ${catkin_LIBRARIES})
add_dependencies(kmean_test ${catkin_EXPORTED_TARGETS})

add_executable(Q_learning src/Q_learning.cpp)
target_link_libraries(Q_learning ${catkin_LIBRARIES})
add_dependencies(Q_learning ${catkin_EXPORTED_TARGETS})
