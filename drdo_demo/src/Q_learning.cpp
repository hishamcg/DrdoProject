#include <cmath> 
#include <termios.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/poll.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <geometry_msgs/Twist.h>
#include <boost/thread/thread.hpp>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

using namespace std;
using namespace cv;
using namespace sensor_msgs;
using namespace message_filters;

int img_x = -1,img_y = -1,th = 15,skip_frame = 0,kfd = 0,numstr = 0,folder_count_F = 0,folder_count_L = 0,folder_count_R = 0,fold_count = 0;
bool rightFirstTime = true,leftFirstTime = true,forwardFirstTime = true,a_pressed = false,b_pressed = false,is_moving = false, check_point = true, reset_bot = true,ROSLoop = true;
float hor_pixel_angle = 0.18125,ver_pixel_angle = 0.18750,x_speed = 0.2,dist_1 = 0,dist_2 = 0,z_angle = 0.2,turn_angle = 10,data_count = 1000;//45/240;//58/320;
float sensor_reward = 0,beta=0.1, beta2=0.08,epsilon = 0.01, Q_gamma = 0.9, alpha = 0.1;
double pi = 3.1415926535897931,one_degree = (2 * pi)/180,old_yaw = 0,rotation_angle = 0,sec_initial,sec_final;
double old_x = 0,old_y = 0,new_x = 0,new_y=0;
string directory = "/home/hisham/iitm_ra/drdo_data/",folder = "set_right" ,name = "/kinect_depth_" ,ext =".bmp";
static const std::string OPENCV_WINDOW = "Image_window";

cv::Vec3b img_intensity;
cv::Size size(320,240);
cv::Mat Kinect_Ind = (cv::Mat_<double>(3,3) << 554.254691191187, 0.0, 160.5, 0.0, 554.254691191187, 120.5, 0.0, 0.0, 1.0);
cv::Mat entityTestMat,Q_sa = Mat::zeros(30,3, CV_32FC1); 

sensor_msgs::ImageConstPtr currentEntity;
struct termios cooked, raw;

#define KEYCODE_A 0x61
#define KEYCODE_B 0x62

image_transport::Subscriber image_sub_;
image_transport::Publisher image_pub_;
geometry_msgs::Twist cmdvel_;
ros::ServiceClient client;
ros::ServiceClient get_client;
gazebo_msgs::SetModelState setmodelstate;
gazebo_msgs::GetModelState getmodelstate;
gazebo_msgs::ModelState modelstate;
//Labrob Position
geometry_msgs::Point labrob_position;
//Labrob orientation
geometry_msgs::Quaternion labrob_orientation;
//Labrob pose (Pose + Orientation)
geometry_msgs::Pose labrob_pose;
ros::Subscriber depth_sub_, sonar1_, sonar2_, sonar3_, sonar4_, sonar5_;
ros::Publisher cmd_pub_;

CvSVM svmForward,svmLeft,svmRight;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Range,
                                                        sensor_msgs::Range,
                                                        sensor_msgs::Range,
                                                        sensor_msgs::Range,
                                                        sensor_msgs::Range> SensorSyncPolicy;
message_filters::Synchronizer<SensorSyncPolicy>* sensor_sync;

void initailizResetState(){
    labrob_position.x = 0.0;
    labrob_position.y = 0.0;
    labrob_position.z = 0.0;

    labrob_orientation.x = 0.0;
    labrob_orientation.y = 0.0;
    labrob_orientation.z = 0.0;
    labrob_orientation.w = 0.0;

    labrob_pose.position = labrob_position;
    labrob_pose.orientation = labrob_orientation;
    modelstate.pose = labrob_pose;
    setmodelstate.request.model_state = modelstate;
}

bool IsNumber(double x){
    return (x == x); 
}

int Q_argmax(int a[]){
    Mat m = (Mat_<float>(3,1) << Q_sa.at<float>(a[0],0),Q_sa.at<float>(a[1],1),Q_sa.at<float>(a[2],2));
    double min, max;
    int minInd, maxInd;
    cv::minMaxIdx(m, &min, &max, &minInd, &maxInd, Mat());
    cout << "===>" << m << endl; 
    return maxInd;
}

float Q_max(int a){
    return max(max(Q_sa.at<float>(a,0),Q_sa.at<float>(a,1)),Q_sa.at<float>(a,2));
}

cv::Mat AngularHistogram(cv::Mat image_o, cv::Mat mat_x, cv::Mat mat_y){

    cv::Mat output;
    int col_step = 32;
    int row_step = 24;
    int rows = image_o.rows;
    int cols = image_o.cols;
    int nimages = 1; // Only 1 image, that is the Mat scene.
    int channels[] = {0}; // Index for hue channel
    int dims = 1; // Only 1 channel, the hue channel
    int histSize[] = {9}; // 9 bins, 1 each for Red, RY, Yellow, YG etc.
    float hranges[] = {0, 181};
    const float *ranges[] = {hranges};
    bool uniform = true; bool accumulate = false;

    for(int j=0; j<cols; j+=col_step){
        for(int i=0; i<rows; i+=row_step){
            //temp.val[j,i];
            cv::Mat sub_image, sub_mat_x, sub_mat_y;
            //cv::Mat hist_x(1, 21, CV_32FC1),hist_y;
            cv::Mat hist_x,hist_y;
            cv::Mat combined;

            image_o(cv::Rect(j,i,col_step,row_step)).copyTo(sub_image);
            mat_x(cv::Rect(j,i,col_step,row_step)).copyTo(sub_mat_x);
            mat_y(cv::Rect(j,i,col_step,row_step)).copyTo(sub_mat_y);

            double sub_min, sub_max, sub_mean;
            cv::minMaxLoc(sub_image, &sub_min, &sub_max);
            sub_mean = cv::mean(sub_image).val[0];
            if(!IsNumber(sub_mean))
                sub_mean = 0;
            calcHist(&sub_mat_x, nimages, channels, Mat(), hist_x, dims, histSize, ranges, uniform, accumulate);
            calcHist(&sub_mat_y, nimages, channels, Mat(), hist_y, dims, histSize, ranges, uniform, accumulate);
            //cout << "min max mean  => " << sub_min << ", " << sub_max <<", "<< sub_mean << endl;  
            //hist_x.push_back({sub_mean,sub_min,sub_max});
            cv::Mat trr = (Mat_<float>(3,1) << sub_mean,sub_min,sub_max);

            vconcat(hist_y,trr,hist_y);
            hconcat(hist_x.t(),hist_y.t(),combined);
            //20 being number of row blocks
            output.push_back(combined);
            //hist_x.copyTo(output.row(i*20+j));
        }
    }

    return output.reshape(1,output.rows * output.cols);
}

cv::Mat normalFeature(const sensor_msgs::ImageConstPtr& msg){

    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);//now cv_ptr is the matrix, do not forget "TYPE_" before "16UC1"
    }catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv_ptr->image;
    }

    cv::Mat image_o;
    
    //TODO:this is not required
    double min_val, max_val;
    cv::minMaxLoc(cv_ptr->image, &min_val, &max_val);

    //ROS_INFO("min = %f , max = %f",min_val,max_val);
    cv::resize(cv_ptr->image,image_o,size);

    //normalFeature(temp_resized);

    cv::Mat mat_x(240, 320, CV_8UC1); 
    cv::Mat mat_y(240, 320, CV_8UC1);

    int cols = image_o.cols,rows = image_o.rows;

    for(int poo=0; poo<cols; poo++){
        for(int joo=0; joo<rows; joo++){

            if(IsNumber(image_o.at<float>(joo,poo))){

                float left,right,top,bottom;

                if(poo == 0){
                    left = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo,poo+1);
                    right = image_o.at<float>(joo,poo+1);
                }else if(poo == cols-1){
                    left = image_o.at<float>(joo,poo-1);
                    right = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo,poo-1);
                }else{
                    left = image_o.at<float>(joo,poo-1);
                    right = image_o.at<float>(joo,poo+1);

                    if(!IsNumber(left))
                        left = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo,poo+1);
                    if(!IsNumber(right))
                        right = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo,poo-1);
                }

                if(joo == 0){
                    top = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo+1,poo);
                    bottom = image_o.at<float>(joo+1,poo);
                }else if(joo == rows-1){
                    top = image_o.at<float>(joo-1,poo);
                    bottom = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo-1,poo);
                }else{
                    top = image_o.at<float>(joo-1,poo);
                    bottom = image_o.at<float>(joo+1,poo);

                    if(!IsNumber(top))
                        top = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo+1,poo);
                    if(!IsNumber(bottom))
                        bottom = 2*image_o.at<float>(joo,poo) - image_o.at<float>(joo-1,poo);
                }

                float gx,gy,hor_pixel_size,ver_pixel_size;
                if(left >= 0.8 && right >= 0.8 && top >= 0.8 && bottom >= 0.8 ){
                    hor_pixel_size = 2*max(left,right)*sin(hor_pixel_angle*pi/180);
                    gx = (left - right) / hor_pixel_size;
                    ver_pixel_size = 2*max(top,bottom)*sin(ver_pixel_angle*pi/180);
                    gy = (top - bottom) / ver_pixel_size;
                }else{
                    gx = 0;gy = 0;
                }

                mat_x.at<uchar>(joo,poo) = (int)(atan(gx) * 180 / pi)+90;
                mat_y.at<uchar>(joo,poo) = (int)(atan(gy) * 180 / pi)+90;

            }else{
                mat_x.at<uchar>(joo,poo) = 255;
                mat_y.at<uchar>(joo,poo) = 255;
            }
            //cout << "left,right,top,bottom,hor,ver,gx,gy = "<< left << "," << right << "," << top << "," << bottom << "," << hor_pixel_size << "," << ver_pixel_size << "," << gx << "," << gy <<endl;
        }
    }
    return AngularHistogram(image_o,mat_x,mat_y);
    //clusterData(FeatureEntity);
    //cout<<FeatureEntity.cols<<" "<<FeatureEntity.rows<<endl;
}

cv::Mat normalizeMD(Mat colVec,Mat me,Mat de){
    cv::Mat colVecD = Mat_<float>(colVec.rows,colVec.cols);
    for(int i=0;i<colVec.cols;i++){
        //subtract(colVec.col(i),me.col(i),colVecD.col(i));
        colVecD.at<float>(0,i) = colVec.at<float>(0,i) - me.at<float>(0,i);
        if(de.at<float>(0,i) > 0){
            colVecD.at<float>(0,i) = colVecD.at<float>(0,i)/de.at<float>(0,i);
        }
    }
    return colVecD;
}

bool checkRobotError(double roll,double pitch){
    if(roll > 0.1 || pitch > 0.1){
        cmdvel_.angular.z = 0;
        cmdvel_.linear.x = 0;
        cmd_pub_.publish(cmdvel_);
        if (client.call(setmodelstate)){
            ROS_INFO("\n##########\nreseting coz of model disorientation\n##########");
            cv::waitKey(20);
        }else{
            ROS_ERROR("Failed to call service ");
        }
        return false;
    }
    return true;
}

bool checkOrientation(){
    tf::Quaternion q(getmodelstate.response.pose.orientation.x, getmodelstate.response.pose.orientation.y, getmodelstate.response.pose.orientation.z, getmodelstate.response.pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return checkRobotError(roll,pitch);
}

void forwardCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::Range::ConstPtr& rang1,const sensor_msgs::Range::ConstPtr& rang2,
                    const sensor_msgs::Range::ConstPtr& rang3,const sensor_msgs::Range::ConstPtr& rang4,const sensor_msgs::Range::ConstPtr& rang5){
    if(get_client.call(getmodelstate)){
        if(checkOrientation()){
            if(forwardFirstTime){
                //initailizing the current state;
                old_x = getmodelstate.response.pose.position.x;
                old_y = getmodelstate.response.pose.position.y;
                //ROS_INFO("x = %lf,y = %lf",old_x,old_y);
                forwardFirstTime = false;
            }

            //ROS_INFO("im stuck");

            new_x = getmodelstate.response.pose.position.x;
            new_y = getmodelstate.response.pose.position.y;
            //ROS_INFO("x = %lf,y = %lf",new_x,new_y);
            float dist = sqrt(pow((new_x-old_x),2) + pow((new_y-old_y),2));
            sec_final = ros::Time::now().toSec();
            //ROS_INFO("distance = %f ", dist);
            if(dist < 0.0001){
                cmdvel_.linear.x = x_speed;
                cmd_pub_.publish(cmdvel_);
                sec_initial = ros::Time::now().toSec();
                ROS_INFO("start forward");
            }else if((sec_final - sec_initial) >= 2){
                //this is the next state
                currentEntity = msg;
                //entityTestMat = normalizeMD(normalFeature(msg).t());
                //take the sensor with min value and compute reward
                //give small forward reward
                sensor_reward = -1*exp(-(std::min(std::min(rang3->range, rang4->range), rang5->range))/beta) + 0.1;
                ROS_INFO("reward f => [%f]",sensor_reward);
                check_point = false;
                dist_1 = dist;  
                cmdvel_.linear.x = 0;
                cmd_pub_.publish(cmdvel_);   
                old_x = new_x;
                old_y = new_y;
                ROSLoop = false;
            }
        }else{
            ROSLoop = false;
        }
    }
}

void leftCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::Range::ConstPtr& rang1,const sensor_msgs::Range::ConstPtr& rang2,
                    const sensor_msgs::Range::ConstPtr& rang3,const sensor_msgs::Range::ConstPtr& rang4,const sensor_msgs::Range::ConstPtr& rang5){

    if(get_client.call(getmodelstate)){
        tf::Quaternion q(getmodelstate.response.pose.orientation.x, getmodelstate.response.pose.orientation.y, getmodelstate.response.pose.orientation.z, getmodelstate.response.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        if(checkRobotError(roll,pitch)){

            if(leftFirstTime){
                old_yaw = yaw;
                leftFirstTime = false;
            }

            if(old_yaw+turn_angle*one_degree > pi && yaw < 0){
                rotation_angle = old_yaw+turn_angle*one_degree - 2*pi;
            }else{
                rotation_angle = old_yaw+turn_angle*one_degree;
            }

            //ROS_INFO("capturing first image - angle = %f , %f, %f", yaw,old_yaw,rotation_angle);

            if(yaw < old_yaw + 0.0001 && yaw > old_yaw - 0.0001){
                cmdvel_.angular.z = z_angle;
                cmd_pub_.publish(cmdvel_);
                ROS_INFO("start left");
            }else if(yaw >= rotation_angle){
                currentEntity = msg;
                sensor_reward = -1*exp(-(rang1->range)/beta2);
                ROS_INFO("reward l => [%f]",sensor_reward);
                cmdvel_.angular.z = 0;
                cmd_pub_.publish(cmdvel_);
                old_yaw = yaw;
                ROSLoop = false;
            }
        }else{
            ROSLoop = false;
        }
    }
}


void rightCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::Range::ConstPtr& rang1,const sensor_msgs::Range::ConstPtr& rang2,
                    const sensor_msgs::Range::ConstPtr& rang3,const sensor_msgs::Range::ConstPtr& rang4,const sensor_msgs::Range::ConstPtr& rang5){
    if(get_client.call(getmodelstate)){
        tf::Quaternion q(getmodelstate.response.pose.orientation.x, getmodelstate.response.pose.orientation.y, getmodelstate.response.pose.orientation.z, getmodelstate.response.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        if(checkRobotError(roll,pitch)){
            if(rightFirstTime){
                //initailizing the current state;
                old_yaw = yaw;
                rightFirstTime = false;
            }

            if(old_yaw-turn_angle*one_degree < -pi && yaw > 0){
                rotation_angle = old_yaw-turn_angle*one_degree + 2*pi;
            }else{
                rotation_angle = old_yaw-turn_angle*one_degree;
            }

            //ROS_INFO("capturing first image - angle = %f , %f, %f", yaw,old_yaw,rotation_angle);

            if(yaw < old_yaw + 0.0001 && yaw > old_yaw - 0.0001){
                cmdvel_.angular.z = -z_angle;
                cmd_pub_.publish(cmdvel_);
                ROS_INFO("start right");
            }else if(yaw <= rotation_angle){
                currentEntity = msg;
                sensor_reward = -1*exp(-(rang2->range)/beta2);
                ROS_INFO("reward r => [%f]",sensor_reward);
                cmdvel_.angular.z = 0;
                cmd_pub_.publish(cmdvel_);
                old_yaw = yaw;
                ROSLoop = false;
            }
        }else{
            ROSLoop = false;
        }
    }
}

void rosLoop(){
    ros::Rate loop_rate(20);
    ROSLoop = true;
    while (ros::ok() && ROSLoop){
        ros::spinOnce();
        loop_rate.sleep();
    }
    ROS_INFO("exit rosloop");
}

void action_forward(ros::NodeHandle node){
    //depth_sub_ = node.subscribe("/camera/depth/image_raw", 1, forwardCallback);
    message_filters::Subscriber<sensor_msgs::Image> img_sub0(node, "/camera/depth/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub1(node, "/sonar_1", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub2(node, "/sonar_2", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub3(node, "/sonar_3", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub4(node, "/sonar_4", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub5(node, "/sonar_5", 1);

    sensor_sync = new message_filters::Synchronizer<SensorSyncPolicy>(SensorSyncPolicy(10),img_sub0, rang_sub1, rang_sub2, rang_sub3, rang_sub4, rang_sub5);
    sensor_sync->registerCallback(boost::bind(&forwardCallback, _1, _2,_3,_4,_5,_6));
    forwardFirstTime = true;
    rosLoop();
}

void action_left(ros::NodeHandle node){
    //depth_sub_ = node.subscribe("/camera/depth/image_raw", 1, leftCallback);
    message_filters::Subscriber<sensor_msgs::Image> img_sub0(node, "/camera/depth/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub1(node, "/sonar_1", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub2(node, "/sonar_2", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub3(node, "/sonar_3", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub4(node, "/sonar_4", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub5(node, "/sonar_5", 1);

    sensor_sync = new message_filters::Synchronizer<SensorSyncPolicy>(SensorSyncPolicy(10),img_sub0, rang_sub1, rang_sub2, rang_sub3, rang_sub4, rang_sub5);
    sensor_sync->registerCallback(boost::bind(&leftCallback, _1, _2,_3,_4,_5,_6));
    leftFirstTime = true;
    rosLoop();
}

void action_right(ros::NodeHandle node){
    //depth_sub_ = node.subscribe("/camera/depth/image_raw", 1, rightCallback);
    message_filters::Subscriber<sensor_msgs::Image> img_sub0(node, "/camera/depth/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub1(node, "/sonar_1", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub2(node, "/sonar_2", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub3(node, "/sonar_3", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub4(node, "/sonar_4", 1);
    message_filters::Subscriber<sensor_msgs::Range> rang_sub5(node, "/sonar_5", 1);

    sensor_sync = new message_filters::Synchronizer<SensorSyncPolicy>(SensorSyncPolicy(10),img_sub0, rang_sub1, rang_sub2, rang_sub3, rang_sub4, rang_sub5);
    sensor_sync->registerCallback(boost::bind(&rightCallback, _1, _2,_3,_4,_5,_6));
    rightFirstTime = true;
    rosLoop();
}

Mat meanDev(String foo,String tag){
    FileStorage fs(foo, FileStorage::READ);
    Mat mat;
    fs[tag] >> mat;
    cout << "loaded effect data. col = "<<mat.cols<<", rows = "<<mat.rows<<endl;
    return mat;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "demo");
    //cv::namedWindow(OPENCV_WINDOW);
    ros::NodeHandle node;
    // image_transport::ImageTransport it_;
    Mat featureMean = meanDev("/home/hisham/iitm_ra/drdo_data/feature_mean_all.yml","Mean");
    Mat featureDev = meanDev("/home/hisham/iitm_ra/drdo_data/feature_deviation_all.yml","Deviation");

    svmForward.load("/home/hisham/iitm_ra/drdo_data/svm_forward.xml");
    svmLeft.load("/home/hisham/iitm_ra/drdo_data/svm_left.xml");
    svmRight.load("/home/hisham/iitm_ra/drdo_data/svm_right.xml");

    client = node.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    get_client = node.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    getmodelstate.request.model_name ="pioneer";
    modelstate.model_name = "pioneer";
    initailizResetState();
    //new addition
    //tf::TransformListener listener;
    cmd_pub_ = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    //first time move
    action_forward(node);

    Mat m_test;
    bool terminate = true;
    int pos_id = 0,currentPos = 0;
    int fw = 0,lf = 0,ri = 0, count = 0;
    while(terminate){
        count++;
        //TODO 
        m_test = normalizeMD(normalFeature(currentEntity).t(),featureMean,featureDev);

        fw = (int)svmForward.predict(m_test.row(0));
        lf = (int)svmLeft.predict(m_test.row(0))+10;
        ri = (int)svmRight.predict(m_test.row(0))+20;

        ROS_INFO("fw => %d, lf => %d, ri => %d",fw,lf,ri);
        int val[] = {fw,lf,ri};
        
        float rndm = rand()/(float)RAND_MAX;

        if(rndm < epsilon){
            pos_id = rand() % 3;
        }else{
            //first 10 values for forward second 10 values for left last 10 vales for right
            pos_id = Q_argmax(val);
        }

        ROS_INFO("pos => %d, val => %d",pos_id,val[pos_id]);

        if(pos_id == 0){
            ROS_INFO("move forward ");
            action_forward(node); 
        }else if(pos_id == 1){
            ROS_INFO("turn left ");
            action_left(node); 
        }else if(pos_id == 2){
            ROS_INFO("turn right ");
            action_right(node);
        } 

        Q_sa.at<float>(currentPos,pos_id) = Q_sa.at<float>(currentPos,pos_id) + alpha*(sensor_reward + Q_gamma*Q_max(val[pos_id]) - Q_sa.at<float>(currentPos,pos_id));
        cout << Q_sa.at<float>(currentPos,pos_id) << endl;
        currentPos = val[pos_id];
        //cout << Q_sa.t() << endl;
        if(count > 1000){
            terminate = false;
        }
        //std::cin.ignore();
    }
    return 0;
}
