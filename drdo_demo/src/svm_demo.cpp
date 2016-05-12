#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

int main(){
	int width = 650, height = 650;
	//Create a mat object
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[10] = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 }; //
	cv::Mat labelsMat(10, 1, CV_32FC1, labels);

	Mat R = Mat(3, 2, CV_8UC3);
    randu(R, Scalar::all(0), Scalar::all(255));
    cout << "R = " << endl << " " << R << endl << endl; 


	float trainingData[10][2] = { { 43, 27 }, { 12, 45 }, { 32, 4 }, { 43, 17 }, { 10, 1 }, {345, 234}, {345, 255}, {374, 276}, {167, 543}, {423, 150} };//
	cv::Mat trainingDataMat(10, 2, CV_32FC1, trainingData);

	// Define up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);
	//SVM.save("/home/hisham/iitm_ra/drdo_data/svm.xml");

	float res = SVM.predict( (cv::Mat_<float>(1, 2) << 10, 10));

	cout<<"predicted => "<< res <<endl;

	cv::Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j){
			cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << j, i);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<cv::Vec3b>(i, j) = green;
			else if (response == 0)
				image.at<cv::Vec3b>(i, j) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, cv::Point( 43, 27 ), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point( 12, 45 ), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point( 32, 4 ), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point( 43, 17 ), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point( 10, 1 ), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(345, 234), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(345, 255), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(374, 276), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(167, 543), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(423, 150), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	circle(image, cv::Point(10, 10), 5, cv::Scalar(0, 0, 0), thickness, lineType);

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	cv::waitKey(0);

	return 0;
}