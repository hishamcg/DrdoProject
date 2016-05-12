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

int kfd = 0;
struct termios cooked, raw;
double pi = 3.1415926535897931;
double one_degree = (2 * pi)/180;
double rotation_angle = 0;
float x_speed = 0.2;  // 0.1 m/s
cv::Mat Kinect_Ind = (cv::Mat_<double>(3,3) << 554.254691191187, 0.0, 160.5, 0.0, 554.254691191187, 120.5, 0.0, 0.0, 1.0);
cv::Mat FeatureOutput; 
string directory = "/home/hisham/iitm_ra/kinect_data/",folder = "set_" ,name = "/kinect_depth_" ,ext =".bmp";
int numstr = 0,folder_count = 0;

#define KEYCODE_A 0x61
#define KEYCODE_B 0x62

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

    gazebo_msgs::SetModelState setmodelstate;
    gazebo_msgs::GetModelState getmodelstate;
    gazebo_msgs::ModelState modelstate;

    //Labrob Position
    geometry_msgs::Point labrob_position;
    //Labrob orientation
    geometry_msgs::Quaternion labrob_orientation;
    //Labrob pose (Pose + Orientation)
    geometry_msgs::Pose labrob_pose;
    
    

    public:
    ImageConverter()
    : it_(nh_){
        // Subscrive to input video feed and publish output video feed
        //image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &ImageConverter::imageCb, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 0, &ImageConverter::imageDepth, this);
        image_pub_ = it_.advertise("/image_converter/output_video", 1);
        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 0);

        labrob_position.x = 0.0;
        labrob_position.y = 0.0;
        labrob_position.z = 0.0;

        labrob_orientation.x = 0.0;
        labrob_orientation.y = 0.0;
        labrob_orientation.z = 0.0;
        labrob_orientation.w = 0.0;

        labrob_pose.position = labrob_position;
        labrob_pose.orientation = labrob_orientation;

        client = nh_.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
        get_client = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
        getmodelstate.request.model_name ="pioneer";
        modelstate.model_name = "pioneer";
    }

    ~ImageConverter(){
        cv::destroyWindow(OPENCV_WINDOW);
        cv::destroyWindow(OPENCV_DEPTH);
    }

    void keyboardLoop();

    bool IsNumber(double x) 
    {
        return (x == x); 
    }

    void normalFeature(cv::Mat image_o){
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
        FeatureOutput.push_back(AngularHistogram(image_o,mat_x,mat_y));
        //clusterData(FeatureOutput);
        //cout<<FeatureOutput.cols<<" "<<FeatureOutput.rows<<endl;
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
        return output.reshape(1, output.rows * output.cols);
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

    void save_image(const sensor_msgs::ImageConstPtr& msg, std::string file){
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

        //saving image
        cv::imwrite(file, temp);
    }

    void imageDepth(const sensor_msgs::ImageConstPtr& msg){
        
        
        if(a_pressed){
            if(reset_bot){
                //labrob_orientation.z = rotation_angle;
                //labrob_orientation.w = rotation_angle;
                labrob_orientation = tf::createQuaternionMsgFromYaw(rotation_angle);
                labrob_pose.orientation = labrob_orientation;
                rotation_angle += one_degree;
                
                modelstate.pose = labrob_pose;
                setmodelstate.request.model_state = modelstate;
                
                if (client.call(setmodelstate)){
                    ROS_INFO("x = %f , rotation_angle =  %f, %f",modelstate.pose.position.x,modelstate.pose.orientation.w,modelstate.pose.orientation.z);
                }else{
                    ROS_ERROR("Failed to call service ");
                    is_moving = false;
                }

                reset_bot = false;
                //cv::waitKey(5);
            }else if(get_client.call(getmodelstate)){
                double dist = sqrt(pow(getmodelstate.response.pose.position.x,2) + pow(getmodelstate.response.pose.position.y,2));
                ROS_INFO("distance = %f , orientation = %f , %f",dist,getmodelstate.response.pose.orientation.z,getmodelstate.response.pose.orientation.w);

                if(dist < 0.001 && !is_moving){
                    numstr = 0;
                    numstr++;
                    std::stringstream out,folder_str;
                    out << numstr;
                    folder_str << folder_count;

                    string folderCreateCommand = "mkdir " + directory + folder + folder_str.str();
                    system(folderCreateCommand.c_str());

                    ROS_INFO("capturing initial image");
                    std::string result = directory + folder + folder_str.str() + name + out.str() + ext;
                    cout << result <<endl;
                    //cv::imwrite(result, temp);
                    save_image(msg, result);
                    //increment pic number
                    numstr++;

                    cmdvel_.linear.x = x_speed;
                    ROS_INFO("About to be moving forward!");
                    cmd_pub_.publish(cmdvel_);

                    is_moving = true;
                }if( dist >= 0.50 && check_point){
                    
                    std::stringstream out,folder_str;
                    out << numstr;
                    folder_str << folder_count;

                    ROS_INFO("capturing second image");
                    std::string result = directory + folder + folder_str.str() + name + out.str() + ext;
                    cout << result <<endl;
                    //cv::imwrite(result, temp);
                    save_image(msg, result);
                    //increment pic number
                    numstr++;

                    check_point = false;
                    
                }else if(dist >= 1.0){
                    std::stringstream out,folder_str;
                    out << numstr;
                    folder_str << folder_count;

                    ROS_INFO("capturing third image");
                    std::string result = directory + folder + folder_str.str() + name + out.str() + ext;
                    cout << result <<endl;
                    //cv::imwrite(result, temp);
                    save_image(msg, result);
                    //increment pic number
                    folder_count++;
                    //stop the robot
                    cmdvel_.linear.x = 0;
                    ROS_INFO("stopping robot");
                    cmd_pub_.publish(cmdvel_);
                    //cv::waitKey(5);

                    check_point = true;
                    is_moving = false;
                    reset_bot = true;
                }
            }
            

        }else if(b_pressed){
            a_pressed = false;
            b_pressed =false;
            cmdvel_.linear.x = 0;
            ROS_INFO("stopping robot");
            cmd_pub_.publish(cmdvel_);
        }

        // Update GUI Window
        //cv::imshow(OPENCV_DEPTH, temp);
        //cv::waitKey(3);

        //int depth = cv_ptr->image.at<short int>(cv::Point(240,320));//you can change 240,320 to your interested pixel
        //ROS_INFO("Depth: %d", depth);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (img_x > -1){
            img_intensity = cv_ptr->image.at<cv::Vec3b>(img_y, img_x);
            cout << "intensity at pos (" << img_x << ", " << img_y << ") = " << img_intensity << endl;
            img_x = -1;
            img_y = -1;
            clicked = true;
        }
        if(clicked){
            int count = 0;
            int centroid_x = 0;
            int centroid_y = 0;

            int rows = cv_ptr->image.rows;
            int cols = cv_ptr->image.cols;

            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){ 
                    cv::Vec3b intensity = cv_ptr->image.at<cv::Vec3b>(i, j);
                    if(intensity.val[0]<img_intensity.val[0]+th && intensity.val[0]>img_intensity.val[0]-th && 
                        intensity.val[1]<img_intensity.val[1]+th && intensity.val[1]>img_intensity.val[1]-th && 
                        intensity.val[2]<img_intensity.val[2]+th && intensity.val[2]>img_intensity.val[2]-th){
                        cv_ptr->image.at<cv::Vec3b>(i, j).val[0] = 255;
                        cv_ptr->image.at<cv::Vec3b>(i, j).val[1] = 255;
                        cv_ptr->image.at<cv::Vec3b>(i, j).val[2] = 255;

                        count++;
                        centroid_y += i;
                        centroid_x += j;
                    }
                }
            }

            // Draw an example circle on the video stream
            if(centroid_y>0 && centroid_x>0 && count>50){
                centroid_x = centroid_x/count;
                centroid_y = centroid_y/count;

                if(centroid_y<rows && centroid_y<cols){
                    cv::circle(cv_ptr->image, cv::Point(centroid_x, centroid_y), 20, CV_RGB(255,0,0));

                    int diff = centroid_x - cols/2;
                    if(std::abs(diff) > 30){
                        if(cmdvel_.angular.z == 0.0){
                            cmdvel_.angular.z = -0.5 * std::abs(diff)/diff;
                            cout << "turn value = " << cmdvel_.angular.z << endl;
                        }
                    }else{
                        if(cmdvel_.angular.z != 0.0){
                            cmdvel_.angular.z = 0.0;
                            cout << "stopped" << endl;
                        }
                    }
                    cmdvel_.linear.x = 0.0;
                    cmd_pub_.publish(cmdvel_);
                }
            }
        }


        // Update GUI Window
        //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        //cv::waitKey(3);

        // Output modified video stream
        //image_pub_.publish(cv_ptr->toImageMsg());
    }
};

void ImageConverter::keyboardLoop()
{
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

void CallBackFunc(int event, int x, int y, int flags, void* userdata){
    if  ( event == cv::EVENT_LBUTTONDOWN ){
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        img_y = y;
        img_x = x;
    }
    // else if  ( event == cv::EVENT_RBUTTONDOWN ){
    //     cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    // }
    // else if  ( event == cv::EVENT_MBUTTONDOWN ){
    //     cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    // }
    // else if ( event == cv::EVENT_MOUSEMOVE ){
    //     cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
    // }    
}

int main(int argc, char** argv){
    ros::init(argc, argv, "image_converter");
    cv::namedWindow(OPENCV_WINDOW);
    cout << "window created" << endl;
    cv::setMouseCallback(OPENCV_WINDOW, CallBackFunc, NULL);
    ImageConverter ic;
    boost::thread t = boost::thread(boost::bind(&ImageConverter::keyboardLoop, &ic));
    
    //ros::spinOnce();
    ros::spin();

    t.interrupt();
    t.join();
    tcsetattr(kfd, TCSANOW, &cooked);
    return 0;
}
