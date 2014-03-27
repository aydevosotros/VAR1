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
	enum SOURCE {bag, camara, kinect};

	PanoramCreator(SOURCE src) :
			it_(nh_) {
		this->first = true;
		switch(src){
		case bag:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCbSift, this);
			break;
		case camara:
			image_sub_ = it_.subscribe("/camera/image_raw", 1,
			&PanoramCreator::imageCbSift, this);
			break;
		case kinect:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCbSift, this);
		default:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCbSift, this);
		}

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

//		imshow("FirstImage", imageColorFirst);
//		imshow("SecondImage", imageColorSecond);

		SiftDescriptorExtractor extractor;

		Mat desciptorsImg1, desciptorsImg2;

		extractor.compute(this->firstImg, keyPointsFirst, desciptorsImg1);
		extractor.compute(this->secondImg, keyPointsSecond, desciptorsImg2);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< std::vector<DMatch> > matches;
//		matcher.match( desciptorsImg1, desciptorsImg2, matches ); // Esto lo haríamos para tal. Pero queremos...
		matcher.knnMatch(desciptorsImg1, desciptorsImg2, matches, 2); // Encontramos 2 más cercanas

		/** Esto es basurilla **/
		double max_dist = 0; double min_dist = 100, second_min = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < desciptorsImg1.rows; i++ ){
			double dist = matches[i][0].distance;
				if(dist < min_dist)
					min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}
//
//		printf("-- Max dist : %f \n", max_dist );
//		printf("-- Second min dist : %f \n", second_min );
		printf("-- Min dist : %f \n", min_dist );
//		printf("-- El mejor por 0.8 es: %f\n", second_min*0.8);

		  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector<DMatch> good_matches;
		for(int i=0; i<desciptorsImg1.rows; i++){ // Esto lo hacemos por separado para poder analizar cuantos quitamos con cada uno
			if(matches[i][0].distance < 0.8*matches[i][1].distance && matches[i][0].distance < 3*min_dist)
				good_matches.push_back(matches[i][0]);
		}

		Mat img_matches;
		drawMatches(this->firstImg, keyPointsFirst, this->secondImg, keyPointsSecond,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		std::stringstream ss;
		ss << "He encontrado "<<good_matches.size()<<" matches decentes";
		ROS_INFO(ss.str().c_str());

		//-- Show detected matches
//		imshow("Good Matches", img_matches);
		std::vector<Point2f> kImg1;
		std::vector<Point2f> kImg2;

		for( int i = 0; i < good_matches.size(); i++ )			{
			//-- Get the keypoints from the good matches
			kImg1.push_back( keyPointsFirst[ good_matches[i].queryIdx ].pt );
			kImg2.push_back( keyPointsSecond[ good_matches[i].trainIdx ].pt );
		}

		Mat H = findHomography( kImg1, kImg2, CV_RANSAC );

		const std::vector<Point2f> points_ant_transformed(keyPointsFirst.size());
		std::vector<Point2f> keypoints_ant_vector(keyPointsFirst.size());
		cv::KeyPoint::convert(keyPointsFirst,keypoints_ant_vector);

		//transformamos los puntos de la imagen anterior
		perspectiveTransform( keypoints_ant_vector, points_ant_transformed, H);

		//creamos una copia de la imagen actual que usaremos para dibujar
		Mat transformed_image;
		cvtColor(this->secondImg, transformed_image, CV_GRAY2BGR);

		//los que esten mas lejos que este parametro se consideran outliers (o que la transformacion está mal calculada)
		//este valor es orientativo, podeis cambiarlo y ajustarlo a los valores
		float distance_threshold=10.0;
		int contdrawbuenos=0;
		int contdrawmalos=0;
		for ( int i =0;i<good_matches.size();i++)
		{
		    int ind        = good_matches.at(i).trainIdx ;
		    int ind_Ant    = good_matches.at(i).queryIdx;

		    cv::Point2f p=        keyPointsSecond.at(ind).pt;
		    cv::Point2f p_ant=    points_ant_transformed[ind_Ant];

		    circle( transformed_image, p_ant, 5, Scalar(255,0,0), 2, 8, 0 ); //ant blue
		    circle( transformed_image, p, 5, Scalar(0,255,255), 2, 8, 0 ); //current yellow

		    Point pointdiff = p - points_ant_transformed[ind_Ant];
		        float distance_of_points=cv::sqrt(pointdiff.x*pointdiff.x + pointdiff.y*pointdiff.y);

		    if(distance_of_points < distance_threshold){ // los good matches se pintan con un circulo verde mas grand
		        contdrawbuenos++;
		        circle( transformed_image, p, 9, Scalar(0,255,0), 2, 8, 0 ); //current red
		    }
		    else{
		        contdrawmalos++;
		        line(transformed_image,p,p_ant,Scalar(0, 0, 255),1,CV_AA);
		    }
		}

		imshow( "transformed", transformed_image );
//		imwrite("/home/antonio/ejemplowrite.png",transformed_image );

		cv::Mat result;

		warpPerspective(this->firstImg, result, H,
				cv::Size(1000, this->firstImg.rows));
		cv::Mat half(result, cv::Rect(0, 0, this->secondImg.cols, this->secondImg.rows));
		this->secondImg.copyTo(half);
		result.copyTo(this->secondImg);
		imshow("Easy Merge Result", this->secondImg);
//		Mat H = findHomography( kImg1, kImg2, CV_RANSAC );
//		Mat output;
//
//		perspectiveTransform( this->secondImg, output, H);
//
//		// Para montar las imagenes
//		double alpha = 0.5; double beta;
//		Mat dst;
//
//		beta = ( 1.0 - alpha );
//		addWeighted( this->firstImg, alpha, output, beta, 0.0, dst);
//
////		hconcat(this->firstImg, this->secondImg, output);
//		imshow("Result", dst);

		cv::waitKey(3);
		imwrite("/home/antonio/result.png",result);
		std::swap(firstImg, secondImg);
	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "image_converter");
	PanoramCreator ic(PanoramCreator::bag);
	ros::spin();
	return 0;
}
