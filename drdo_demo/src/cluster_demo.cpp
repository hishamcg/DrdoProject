#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/ml/ml.hpp"
#include <ros/ros.h>
#include <fstream>

using namespace std;
using namespace cv;

String file_svm = "/home/hisham/iitm_ra/drdo_data/svm_",file_entity = "/home/hisham/iitm_ra/drdo_data/entity_",file_effect = "/home/hisham/iitm_ra/drdo_data/effect_";
String file_entity_raw = "/home/hisham/iitm_ra/drdo_data/raw_entity_",file_effect_raw = "/home/hisham/iitm_ra/drdo_data/raw_effect_";
String ext_xml = ".xml",ext_yml = ".yml";
string directory = "/home/hisham/iitm_ra/drdo_data/";

Mat clusterData(cv::Mat colVec){
	Mat bestLabels, centers, clustered, colVecNorm;
	int iteration = 500;
	int clusts = 10;
	double eps = 0.001;

	double compactness = kmeans(colVec, clusts, bestLabels, 
	    TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, iteration, eps), 
	    iteration, KMEANS_PP_CENTERS, centers);
	//Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels
	cout << "Compactness = " << compactness << ", size of centers = "<<centers.cols<<" ,"<<centers.rows<< endl;
	return bestLabels;
}

void learnSVM(Mat labels, Mat entity_mat, String file_name){
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 5000, 1e-6);

	CvSVM svm;
	String f_svm = file_svm+file_name+ext_xml;
	svm.train(entity_mat, labels, cv::Mat(), cv::Mat(), params);
	svm.save(f_svm.c_str());
}

void loadFiles(String file_name){
	String f_eff = file_effect+file_name+ext_yml;
	FileStorage fs(f_eff, FileStorage::READ);
	Mat effect_mat;
	fs["Effect"] >> effect_mat;
	cout << "loaded effect data. col = "<<effect_mat.cols<<", rows = "<<effect_mat.rows<<endl; 
	
	Mat labels = clusterData(effect_mat);

	String f_ent = file_entity+file_name+ext_yml;
	FileStorage fs2(f_ent, FileStorage::READ);
	Mat entity_mat;
	fs2["Entity"] >> entity_mat;
	cout << "loaded entity data. col = "<<entity_mat.cols<<", rows = "<<entity_mat.rows<<endl; 
	
	ROS_INFO("Starting SVM for");
	learnSVM(labels,entity_mat,file_name);
}

void save_file(String f_n,cv::Mat Mean,cv::Mat Deviation){
	cout << "loaded mean data. col = "<<Mean.cols<<", rows = "<<Mean.rows<<endl; 
	cout << "loaded deviation data. col = "<<Deviation.cols<<", rows = "<<Deviation.rows<<endl; 

    std::string loc = directory +"feature_mean_"+f_n+ext_yml;
    cv::FileStorage file_raw(loc, cv::FileStorage::WRITE);
    // Write to file!
    file_raw << "Mean" << Mean;
    file_raw.release();

    loc = directory +"feature_deviation_"+f_n+ext_yml;
    cv::FileStorage file(loc, cv::FileStorage::WRITE);
    // Write to file!
    file << "Deviation" << Deviation;
    file.release();
    ROS_INFO("\n\n saved\n\n");
}


void normalizeMD(cv::Mat colVec,String foo){
    cv::Mat mean_mat,dev_mat;
    for(int i=0;i<colVec.cols;i++){
        cv::Scalar mean1, stddev1;
        cv::meanStdDev(colVec.col(i), mean1, stddev1);
        mean_mat.push_back((float)mean1.val[0]);
        dev_mat.push_back((float)stddev1.val[0]);
    }
    //cout << mean_mat << "\n\n\n\n\n"<<dev_mat<<endl;
    save_file(foo,mean_mat.t(),dev_mat.t());
}

void meanDev(){
	Mat all_val,emat;

	FileStorage fs("/home/hisham/iitm_ra/drdo_data/raw_entity_forward.yml", FileStorage::READ);
	fs["Entity"] >> emat;
	all_val.push_back(emat);

	FileStorage fs1("/home/hisham/iitm_ra/drdo_data/raw_entity_left.yml", FileStorage::READ);
	fs1["Entity"] >> emat;
	all_val.push_back(emat);

	FileStorage fs2("/home/hisham/iitm_ra/drdo_data/raw_entity_right.yml", FileStorage::READ);
	fs2["Entity"] >> emat;
	all_val.push_back(emat);

	cout << "loaded effect data. col = "<<all_val.cols<<", rows = "<<all_val.rows<<endl; 
	
	normalizeMD(all_val,"all");
}

void writeCSV(string filename, Mat m){
   ofstream myfile;
   myfile.open(filename.c_str());
   myfile<< cv::format(m, "csv") << std::endl;
   myfile.close();
}

void convertToCsv(){
	Mat fo,le,ri;

	FileStorage fs("/home/hisham/iitm_ra/drdo_data/entity_forward.yml", FileStorage::READ);
	fs["Entity"] >> fo;
	ROS_INFO("saved");

	// FileStorage fs1("/home/hisham/iitm_ra/drdo_data/entity_left.yml", FileStorage::READ);
	// fs1["Entity"] >> le;

	// FileStorage fs2("/home/hisham/iitm_ra/drdo_data/entity_right.yml", FileStorage::READ);
	// fs2["Entity"] >> ri;
	writeCSV("/home/hisham/iitm_ra/drdo_data/csv/ef.csv",fo);

	// FileStorage file1("/home/hisham/iitm_ra/drdo_data/csv/ef.csv", FileStorage::WRITE);
	// std::stringstream ss1;
	// ss1 << format(fo,"csv") << endl << endl;
	// file1 << ss1.str();

	// FileStorage file2("/home/hisham/iitm_ra/drdo_data/csv/ef.csv", FileStorage::WRITE);
	// std::stringstream ss2;
	// ss2 << format(fo,"csv") << endl << endl;
	// file2 << ss2.str();
}

int main(int argc , char** argv){
	// ROS_INFO("MeanDev on all...");
	// meanDev();

	convertToCsv();	

	// ROS_INFO("Learning on Forwrad...");
	// loadFiles("forward");
	// ROS_INFO("Learning on Left...");
	// loadFiles("left");
	// ROS_INFO("Learning on Right...");
	// loadFiles("right");
	return 0;
}
