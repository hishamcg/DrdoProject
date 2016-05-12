#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace std;
using namespace cv;

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

void clusterData(cv::Mat colVec){
	Mat bestLabels, centers, clustered, colVecNorm;
	int iteration = 100;
	int clusts = 3;
	double eps = 0.001;
	//cv::Mat temp = (Mat_<float>(3,3) << 0.1,100,10000,0.2,200,20000,0.9,900,90000);
	//colVecNorm = normalizeMD(colVec);
	//cout << "value  " << colVec.col(1)<<endl;
	//cout << "start kmean " << colVecNorm.col(1)<<endl;

	double compactness = kmeans(colVec, clusts, bestLabels, 
	    TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, iteration, eps), 
	    iteration, KMEANS_PP_CENTERS, centers);
	//Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels
	cout << "Compactness = " << compactness << ", size of centers = "<<centers.cols<<" ,"<<centers.rows<< endl;
	cout << "size of bestLabels = 	"<<bestLabels<< endl;

}

int main(int argc , char** argv){
	FileStorage fs2("/home/hisham/iitm_ra/kinect_data/set_effect_temp.yml", FileStorage::READ);
	Mat cameraMatrix2;
	fs2["EffectMat"] >> cameraMatrix2;

	cout << "woho got it done cho shimble col = "<<cameraMatrix2.cols<<", rows = "<<cameraMatrix2.rows<<endl; 

	clusterData(cameraMatrix2);
	return 0;
}
