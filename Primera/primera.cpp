#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <cmath>

static const std::string OPENCV_WINDOW = "Image window";

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class PanoramCreator {
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	// Propias
	bool first;
	Mat firstImg;
	Mat secondImg;

public:
	PanoramCreator() :
			it_(nh_) {
		this->first = true;
		// Subscrive to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
				&PanoramCreator::imageCbSift, this);
//    image_pub_ = it_.advertise("/image_converter/output_video", 1);

	}

	~PanoramCreator() {
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCbSift(const sensor_msgs::ImageConstPtr& msg) {
		cv_bridge::CvImagePtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
		} catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		vector<KeyPoint> keyPoints;

		Mat imageColor;
		Mat src_gray;

		if (this->first) {
			cvtColor(cv_ptr->image, this->firstImg, CV_BGR2GRAY);
			this->first = false;
			ROS_INFO("He obtenido la primera imagen y espero...");
		} else {
			cvtColor(cv_ptr->image, this->secondImg, CV_BGR2GRAY);
			ROS_INFO("He obtenido nuevas y proceso...");
			this->processPanoram();
		}


	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg) {
		cv_bridge::CvImagePtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
		} catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		std::cerr << " imagecb: " << msg->header.frame_id << " : "
				<< msg->header.seq << " : " << msg->header.stamp << std::endl;

		cv::Mat src_gray;

		cvtColor(cv_ptr->image, src_gray, CV_BGR2GRAY);

		imshow("BN-Image", src_gray);
		Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");
		std::vector<cv::KeyPoint> points;
		detector->detect(src_gray, points);

		cv::Mat imageColor;
		cvtColor(src_gray, imageColor, CV_GRAY2BGR);

		for (size_t i = 0; i < points.size(); i++) {
			circle(imageColor, points[i].pt, 3, CV_RGB(255, 0, 0));
		}
		//se pueden pintar tambien con esta funcion
		//drawKeypoints(imageColor, points, imageColor, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

		imshow("Fast keypoints", imageColor);

		cv::waitKey(3);

	}

	void processPanoram() {
		Mat mask;
		vector<KeyPoint> keyPointsFirst;
		vector<KeyPoint> keyPointsSecond;
		Mat imageColorFirst;
		Mat imageColorSecond;

		cv::FeatureDetector* detector;
		detector = new SiftFeatureDetector(
				0, // nFeatures
				4, // nOctaveLayers
				0.04, // contrastThreshold
				10, //edgeThreshold
				1.6 //sigma
				);

		detector->detect(this->firstImg, keyPointsFirst, mask);
		detector->detect(this->secondImg, keyPointsSecond, mask);

		cvtColor(this->firstImg, imageColorFirst, CV_GRAY2BGR);
		cvtColor(this->secondImg, imageColorSecond, CV_GRAY2BGR);

		for (size_t i = 0; i < keyPointsFirst.size(); i++) {
			circle(imageColorFirst, keyPointsFirst[i].pt, 3, CV_RGB(255, 0, 0));
		}
		for (size_t i = 0; i < keyPointsSecond.size(); i++) {
			circle(imageColorSecond, keyPointsSecond[i].pt, 3, CV_RGB(255, 0, 0));
		}

		imshow("FirstImage", imageColorFirst);
		imshow("SecondImage", imageColorSecond);

		std::stringstream ss;
		ss << "He encontrado "<<keyPointsFirst.size()<<" en la primera";
		ROS_INFO(ss.str().c_str());
		std::stringstream ss2;
		ss2 << "He encontrado "<<keyPointsSecond.size()<<" en la segunda";
		ROS_INFO(ss2.str().c_str());
		cv::waitKey(3);

		std::swap(firstImg, secondImg);
	}

	double euclideanDistance(KeyPoint kp1, KeyPoint kp2){
		double dist;
//		for(int i=0; i<kp1.size; i++){
//			dist+=std::sqrt(std::pow(kp1.pt[i]-kp2.pt[i],2));
//		}

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "image_converter");
	PanoramCreator ic;
	ros::spin();
	return 0;
}
