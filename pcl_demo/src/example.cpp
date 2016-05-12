#include <iostream>
#include <termios.h>
#include <signal.h>
#include <sys/poll.h>
#include <boost/bind.hpp>

#include <ros/ros.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/openni_grabber.h>

#include <boost/thread/thread.hpp>

#define KEYCODE_A 0x61
#define KEYCODE_B 0x62


bool a_pressed = false;
int kfd = 0;
struct termios cooked, raw;

ros::Publisher pub;


void cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud)
{
	pcl::PCLPointCloud2 cloud_filtered;

	// Perform the actual filtering
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud (cloud);
	sor.setLeafSize (0.1, 0.1, 0.1);
	sor.filter (cloud_filtered);

	// Publish the data
	pub.publish (cloud_filtered);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_demo");
  ros::NodeHandle nh;

  //SimpleOpenNIViewer ni;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("input", 1,cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<pcl::PCLPointCloud2> ("output", 1);

  // Spin
  ros::spin ();
}
