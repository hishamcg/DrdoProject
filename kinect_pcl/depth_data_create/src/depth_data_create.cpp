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
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <geometry_msgs/Twist.h>
#include <boost/thread/thread.hpp>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_DEPTH = "Image depth";
using namespace std;
using namespace cv;
int img_x = -1,img_y = -1;
cv::Vec3b img_intensity;
int th = 15;
bool clicked = false,a_pressed = false,b_pressed = false,is_moving = false, check_point = true, reset_bot = true;
cv::Size size(320,240);

float hor_pixel_angle = 0.18125;//58/320;
float ver_pixel_angle = 0.18750;//45/240;

sensor_msgs::ImageConstPtr image_1, image_2, image_3;

int kfd = 0;
struct termios cooked, raw;
double pi = 3.1415926535897931;
double one_degree = pi/180;
double rotation_angle = 0.00001;
float x_speed = 0.2;  // 0.1 m/s
float z_angle = 0.2;
cv::Mat Kinect_Ind = (cv::Mat_<double>(3,3) << 554.254691191187, 0.0, 160.5, 0.0, 554.254691191187, 120.5, 0.0, 0.0, 1.0);

cv::Mat FeatureEntityLeft,EffectMatLeft,FeatureEntityRight,EffectMatRight; 

string directory = "/home/hisham/iitm_ra/kinect_data/",folder = "set_" ,name = "/kinect_depth_" ,ext =".bmp";
int numstr = 0,folder_count = 0;

#define KEYCODE_A 0x61
#define KEYCODE_B 0x62

double sec_initial,sec_final;
float dist_1 = 0,dist_2 = 0;

class ImageConverter{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Publisher image_pub_;
    geometry_msgs::Twist cmdvel_;
    ros::Publisher cmd_pub_;

    ros::ServiceClient client;
    ros::ServiceClient get_client;

    gazebo_msgs::SetModelState setmodelstate, setCylinder1, setCylinder2, setSphere2;
    gazebo_msgs::GetModelState getmodelstate;
    gazebo_msgs::ModelState modelstate;

    //Labrob Position
    geometry_msgs::Point labrob_position;
    //Labrob orientation
    geometry_msgs::Quaternion labrob_orientation;
    //Labrob pose (Pose + Orientation)
    geometry_msgs::Pose labrob_pose,unit_cylinder_1,unit_cylinder_2,unit_sphere_2;

    
    

    public:
    ImageConverter()
    : it_(nh_){
        // Subscrive to input video feed and publish output video feed
        //image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &ImageConverter::imageCb, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &ImageConverter::imageDepth, this);
        image_pub_ = it_.advertise("/image_converter/output_video", 1);
        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        labrob_position.x = 0.0;
        labrob_position.y = 0.0;
        labrob_position.z = 0.0;

        labrob_orientation.x = 0.0;
        labrob_orientation.y = 0.0;
        labrob_orientation.z = 0.0;
        labrob_orientation.w = 0.0;

        labrob_pose.position = labrob_position;
        labrob_pose.orientation = labrob_orientation;
        getmodelstate.request.model_name ="pioneer";
        modelstate.model_name = "pioneer";

        unit_cylinder_1.position.x = 2.132750; unit_cylinder_1.position.y = -0.080122; unit_cylinder_1.position.z = 0.400000;
        unit_cylinder_1.orientation = tf::createQuaternionMsgFromRollPitchYaw(-1.570796,0,0);
        setCylinder1.request.model_state.model_name = "unit_cylinder_1_0";
        setCylinder1.request.model_state.pose = unit_cylinder_1;


        unit_cylinder_2.position.x = -1.811470; unit_cylinder_2.position.y = 0.583056; unit_cylinder_2.position.z = 0.400000;
        unit_cylinder_2.orientation = tf::createQuaternionMsgFromRollPitchYaw(0,0,0);
        setCylinder2.request.model_state.model_name = "unit_cylinder_2";
        setCylinder2.request.model_state.pose = unit_cylinder_2;

        unit_sphere_2.position.x = -1.114410; unit_sphere_2.position.y = -1.618990; unit_sphere_2.position.z = 0.400000;
        unit_sphere_2.orientation = tf::createQuaternionMsgFromRollPitchYaw(0,0,0);
        setSphere2.request.model_state.model_name = "unit_sphere_2_0";
        setSphere2.request.model_state.pose = unit_sphere_2;

        client = nh_.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
        get_client = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    }

    ~ImageConverter(){
        //cv::destroyWindow(OPENCV_WINDOW);
        cv::destroyWindow(OPENCV_DEPTH);
    }

    void keyboardLoop(){
        char c;
        // get the console in raw mode
        tcgetattr(kfd, &cooked);
        memcpy(&raw, &cooked, sizeof(struct termios));
        raw.c_lflag &=~ (ICANON | ECHO);
        raw.c_cc[VEOL] = 1;
        raw.c_cc[VEOF] = 2;
        tcsetattr(kfd, TCSANOW, &raw);
        
        struct pollfd ufd;
        ufd.fd = kfd;
        ufd.events = POLLIN;
        
        for(;;)
        {
            boost::this_thread::interruption_point();
            // get the next event from the keyboard
            int num;
            
            if ((num = poll(&ufd, 1, 250)) < 0){
                perror("poll():");
                return;
            }else if(num > 0){
                if(read(kfd, &c, 1) < 0){
                    perror("read():");
                    return;
                }
            }

            if (KEYCODE_A == c){
                puts("Got A");
                a_pressed = true;
                b_pressed = false;
                c = 0;
            }else if(KEYCODE_B == c){
                puts("Got B");
                a_pressed = false;
                b_pressed = true;
                c = 0;
            }
        }
    }

    bool IsNumber(double x) 
    {
        return (x == x); 
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

    cv::Mat AngularHistogram(cv::Mat image_o, cv::Mat mat_x, cv::Mat mat_y){

        cv::Mat output;
        int col_step = 16;
        int row_step = 12;
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

    void clusterData(cv::Mat colVec){
          Mat bestLabels, centers, clustered;
          int iteration = 20;
          int clusts = 10;
          double eps = 0.001;
          //colVec.convertTo(colVecD, CV_32FC3, 1.0/255.0); // convert to floating point
          double compactness = kmeans(colVec, clusts, bestLabels, 
                TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, iteration, eps), 
                iteration, KMEANS_PP_CENTERS, centers);
          //Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels
          cout << "Compactness = " << compactness << ", size of centers = "<<centers.cols<<" ,"<<centers.rows<< endl;
    }

    void save_image(const sensor_msgs::ImageConstPtr& msg){

        cv_bridge::CvImageConstPtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);//now cv_ptr is the matrix, do not forget "TYPE_" before "16UC1"
        }catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat temp,temp_resized;
        double min_val, max_val;
        cv::minMaxLoc(cv_ptr->image, &min_val, &max_val);

        //ROS_INFO("min = %f , max = %f",min_val,max_val);
        cv::resize(cv_ptr->image,temp_resized,size);

        //no normalization required;
        temp_resized.convertTo(temp, CV_8U, 255.0/(max_val - min_val), -min_val * 255.0/(max_val - min_val));

        //normalFeature(temp_resized);

        std::stringstream out,folder_str;
        out << numstr;
        folder_str << folder_count;

        string folderCreateCommand = "mkdir " + directory + folder + folder_str.str();
        system(folderCreateCommand.c_str());

        std::string result = directory + folder + folder_str.str() + name + out.str() + ext;
        //std::string result = directory + name + out.str() + ext;
        cout << result <<endl;
        //saving image
        cv::imwrite(result, temp);
    }

    void create_folder(){

    }

    cv::Mat normalizeMD(cv::Mat colVec){

        cv::Mat colVecD = (Mat_<float>(colVec.rows,colVec.cols));

        for(int i=0;i<colVec.cols;i++){
            cv::Scalar mean1, stddev1;
            cv::meanStdDev(colVec.col(i), mean1, stddev1);

            //cout << mean1 << ", "<<stddev1<<endl;
            subtract(colVec.col(i),mean1.val[0],colVecD.col(i));
            if(stddev1.val[0] != 0)
                colVecD.col(i) /= stddev1.val[0];
        }
        
        return colVecD;
    }

    void get_featureIn(const sensor_msgs::ImageConstPtr& image_1, const sensor_msgs::ImageConstPtr& image_2){
        cv::Mat a = normalFeature(image_1).t();
        cv::Mat b = normalFeature(image_2).t();
        FeatureEntityLeft.push_back(a);
        FeatureEntityRight.push_back(b);

        cv::Mat subLeft,subRight;

        subtract(b,a,subLeft);
        cv::Mat rob_loc_left = (Mat_<float>(1,3) << 0,0,-45);
        hconcat(subLeft,rob_loc_left,subLeft);
        EffectMatLeft.push_back(subLeft);

        subtract(b,a,subRight);
        cv::Mat rob_loc_right = (Mat_<float>(1,3) << 0,0,45);
        hconcat(subRight,rob_loc_right,subRight);
        EffectMatRight.push_back(subRight);
        //cout<<FeatureEntity.rows << "," <<FeatureEntity.cols << "," <<EffectMat.rows <<","<<EffectMat.cols<<","<<endl;
        ROS_INFO("count -> %d",folder_count);
        if(folder_count>359 ){
            stop_bot();
            std::string loc = directory + folder + "feature_rotate_45_left.yml";
            cv::FileStorage file(loc, cv::FileStorage::WRITE);
            // Write to file!
            file << "FeatureEntity" << normalizeMD(FeatureEntityLeft);
            file.release();

            // Declare what you need
            loc = directory + folder + "effect_rotate_45_left.yml";
            cv::FileStorage file2(loc, cv::FileStorage::WRITE);
            // Write to file!
            cout << EffectMatLeft.cols << " , " << EffectMatLeft.rows;
            file2 << "EffectMat" << normalizeMD(EffectMatLeft);
            file2.release();

            loc = directory + folder + "feature_rotate_45_right.yml";
            cv::FileStorage file3(loc, cv::FileStorage::WRITE);
            // Write to file!
            file3 << "FeatureEntity" << normalizeMD(FeatureEntityRight);
            file3.release();

            // Declare what you need
            loc = directory + folder + "effect_rotate_45_right.yml";
            cv::FileStorage file4(loc, cv::FileStorage::WRITE);
            // Write to file!
            cout << EffectMatRight.cols << " , " << EffectMatRight.rows;
            file4 << "EffectMat" << normalizeMD(EffectMatRight);
            file4.release();
        }
    }

    void stop_bot(){
        a_pressed = false;
        b_pressed =false;
        cmdvel_.linear.x = 0;
        cmdvel_.angular.z = 0;
        ROS_INFO("stopping robot");
        cmd_pub_.publish(cmdvel_);
    }


    void imageDepth(const sensor_msgs::ImageConstPtr& msg){
        if(a_pressed){
            if(reset_bot){
                //labrob_orientation.z = rotation_angle;
                //labrob_orientation.w = rotation_angle;
                labrob_orientation = tf::createQuaternionMsgFromYaw(rotation_angle);
                //cout<<labrob_orientation.x<<","<<labrob_orientation.y<<","<<labrob_orientation.z<<","<<labrob_orientation.w<<endl;
                labrob_pose.orientation = labrob_orientation;
                
                modelstate.pose = labrob_pose;
                setmodelstate.request.model_state = modelstate;
                
                if (client.call(setmodelstate) && client.call(setCylinder1) && client.call(setCylinder2) && client.call(setSphere2)){
                    ROS_INFO("reseting");
                    rotation_angle += one_degree;
                    reset_bot = false;
                    cv::waitKey(200);
                    //ROS_INFO("x = %f , rotation_angle =  %f, %f",modelstate.pose.position.x,modelstate.pose.orientation.w,modelstate.pose.orientation.z);
                }else{
                    ROS_ERROR("Failed to call service ");
                    reset_bot = true;
                }
                //cv::waitKey(5);
            }else if(get_client.call(getmodelstate)){
                //float dist = sqrt(pow(getmodelstate.response.pose.position.x,2) + pow(getmodelstate.response.pose.position.y,2));
                //ROS_INFO("distance = %f , orientation = %f , %f",dist,getmodelstate.response.pose.orientation.z,getmodelstate.response.pose.orientation.w);
                //float rotate_a = tf::getYaw(getmodelstate.response.pose.orientation);
                tf::Quaternion q(getmodelstate.response.pose.orientation.x, getmodelstate.response.pose.orientation.y, getmodelstate.response.pose.orientation.z, getmodelstate.response.pose.orientation.w);
                tf::Matrix3x3 m(q);
                double roll, pitch, yaw;
                m.getRPY(roll, pitch, yaw);
                //
                //sec_final = ros::Time::now().toSec();

                if(yaw < 0 || (rotation_angle > pi+pi/8 && yaw <pi)){
                    yaw = 2*pi+yaw;
                }

                ROS_INFO("capturing first image - angle = %f , %f,", yaw,rotation_angle);

                if(yaw < rotation_angle + 0.001){
                    //set pic number to 0
                    numstr = 0;
                    image_1 = msg;
                    //save_image(msg);

                    cmdvel_.angular.z = z_angle;
                    cmd_pub_.publish(cmdvel_);

                    sec_initial = ros::Time::now().toSec();
                //}if( dist >= 0.50 && check_point){
                }else if(yaw >= 45*one_degree + rotation_angle){

                    cmdvel_.angular.z = 0;
                    cmd_pub_.publish(cmdvel_);
                    image_2 = msg;
                    numstr++;
                    //save_image(msg);
                    folder_count++;

                    ROS_INFO("capturing second image");

                    get_featureIn(image_1, image_2);

                    reset_bot = true;

                }
            }
        }else if(b_pressed){
            stop_bot();
        }

        // Update GUI Window
        //cv::imshow(OPENCV_DEPTH, temp);
        //cv::waitKey(3);

        //int depth = cv_ptr->image.at<short int>(cv::Point(240,320));//you can change 240,320 to your interested pixel
        //ROS_INFO("Depth: %d", depth);
    }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "image_converter");
    //cv::namedWindow(OPENCV_WINDOW);
    cout << "starting" << endl;
    ImageConverter ic;
    boost::thread t = boost::thread(boost::bind(&ImageConverter::keyboardLoop, &ic));
    
    ros::spin();


    t.interrupt();
    t.join();
    tcsetattr(kfd, TCSANOW, &cooked);
}
