#include <geometry_msgs/Twist.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;
void processCloud( const sensor_msgs::PointCloud2ConstPtr& msg ) {
  //ROS_INFO("I heard: [%s]", cloud->data.c_str());
  ROS_INFO("I am inside");
  std::cout << "Received cloud msg: " << "header=" << msg->header << "width="<<msg->width <<", height="<<msg->height<<".\n";
}

int main(int argc, char **argv){

	ros::init(argc, argv, "ball_tracker");

	ros::NodeHandle n;

	ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points", 1, processCloud);

	ros::spin();

    return 0;
}
