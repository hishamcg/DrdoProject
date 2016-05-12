#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/ml/ml.hpp"
#include <ros/ros.h>

using namespace std;
using namespace cv;

String file_svm = "/home/hisham/iitm_ra/drdo_data1/svm_",file_entity = "/home/hisham/iitm_ra/drdo_data1/entity_",file_effect = "/home/hisham/iitm_ra/drdo_data1/effect_";
String ext_xml = ".xml",ext_yml = ".yml";
CvSVM svm;
int test_set_size = 0;

void learnSVM(Mat labels, Mat entity_mat, String file_name){
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 5000, 1e-6);

    String f_svm = file_svm+file_name+ext_xml;
    svm.train(entity_mat, labels, cv::Mat(), cv::Mat(), params);
    svm.save(f_svm.c_str());
}

void testSVM(Mat labels, Mat entity_mat){
    float response;
    int count = 0;
    for(int i = 0;i<entity_mat.rows;i++){
        response = svm.predict(entity_mat.row(i));
        cout << "actual = "<< labels.at<float>(0,i)<<", predicted = "<<response<<endl; 

        if(labels.at<float>(0,i) == response)
            count++;
    }
    cout << "\n\nprediction ratio => " << count <<endl;
}

void loadFiles(String file_name){
    String f_eff = file_effect+file_name+ext_yml;
    FileStorage fs(f_eff, FileStorage::READ);
    Mat effect_mat;
    fs["Effect"] >> effect_mat;
    cout << "loaded effect data. col = "<<effect_mat.cols<<", rows = "<<effect_mat.rows<<endl; 
    
    String f_ent = file_entity+file_name+ext_yml;
    FileStorage fs2(f_ent, FileStorage::READ);
    Mat entity_mat;
    fs2["Entity"] >> entity_mat;
    cout << "loaded entity data. col = "<<entity_mat.cols<<", rows = "<<entity_mat.rows<<endl; 
    
    ROS_INFO("Starting SVM for");

    Mat train_lab,test_lab,train_ent,test_ent;

    cout << "effect_mat size. col = "<<effect_mat.cols<<", rows = "<<effect_mat.rows<<endl; 

    effect_mat(cv::Rect(0,0,effect_mat.cols,effect_mat.rows-test_set_size)).copyTo(train_lab);
    effect_mat(cv::Rect(0,effect_mat.rows-test_set_size,effect_mat.cols,test_set_size)).copyTo(test_lab);

    cout << "loaded train_lab data. col = "<<train_lab.cols<<", rows = "<<train_lab.rows<<endl; 
    cout << "loaded test_lab data. col = "<<test_lab.cols<<", rows = "<<test_lab.rows<<endl; 

    entity_mat(cv::Rect(0,0,entity_mat.cols,entity_mat.rows-test_set_size)).copyTo(train_ent);
    entity_mat(cv::Rect(0,entity_mat.rows-test_set_size,entity_mat.cols,test_set_size)).copyTo(test_ent);

    cout << "loaded train_ent data. col = "<<train_ent.cols<<", rows = "<<train_ent.rows<<endl; 
    cout << "loaded test_ent data. col = "<<test_ent.cols<<", rows = "<<test_ent.rows<<endl; 

    //learnSVM(labels,entity_mat,file_name);
    learnSVM(train_lab,train_ent,file_name);
    //testSVM(test_lab,test_ent);
}

int main(int argc , char** argv){
    ROS_INFO("Learning on Forwrad...");
    loadFiles("forward");
    ROS_INFO("Learning on Left...");
    loadFiles("left");
    ROS_INFO("Learning on Right...");
    loadFiles("right");
}
