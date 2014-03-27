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
#include <exception>

static const std::string OPENCV_WINDOW = "Image window";

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class PanoramCreator {
public:
	enum SOURCE {bag, camera, kinect};
	enum DETECTOR {SIFT_D, SURF_D, FAST, MSER, ORB_D};
	enum EXTRACTOR {SIFT_E, SURF_E, BRIEF, ORB_E};

	PanoramCreator(SOURCE src, DETECTOR dtc, EXTRACTOR ext, float filterValue1, float filterValue2 ) : it_(nh_) {
		this->first = true;
		this->filter1 = filterValue1;
		this->filter2 = filterValue2;
		this->detector = dtc;
		this->extractor = ext;

		switch(src){
		case bag:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
			break;
		case camera:
			image_sub_ = it_.subscribe("/camera/image_raw", 1,
			&PanoramCreator::imageCb, this);
			break;
		case kinect:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
			break;
		default:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
		}

	}

	~PanoramCreator() {
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg) {
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
			this->firstImgColor = cv_ptr->image;
			ROS_INFO("He obtenido la primera imagen y espero...");
		} else {
			cvtColor(cv_ptr->image, this->secondImg, CV_BGR2GRAY);
			this->secondImgColor = cv_ptr->image;
			ROS_INFO("He obtenido nuevas y proceso...");
			this->processPanoram();
		}


	}


	void processPanoram() {
		Mat mask;
		vector<KeyPoint> keyPointsFirst;
		vector<KeyPoint> keyPointsSecond;
		Mat imageColorFirst;
		Mat imageColorSecond;

		cv::FeatureDetector* detector;

		switch(this->detector){
		case SIFT_D:
			detector = new SiftFeatureDetector();
			break;
		case SURF_D:
			detector = new SurfFeatureDetector();
			break;
		case FAST:
			detector = new FastFeatureDetector();
			break;
		case MSER:
			detector = new MserFeatureDetector();
			break;
		case ORB_D:
			detector = new OrbFeatureDetector();
			break;
		default:
			detector = new OrbFeatureDetector();
		}

		detector->detect(this->firstImg, keyPointsFirst, mask);
		detector->detect(this->secondImg, keyPointsSecond, mask);


		for (size_t i = 0; i < keyPointsFirst.size(); i++) {
			circle(imageColorFirst, keyPointsFirst[i].pt, 3, CV_RGB(255, 0, 0));
		}
		for (size_t i = 0; i < keyPointsSecond.size(); i++) {
			circle(imageColorSecond, keyPointsSecond[i].pt, 3, CV_RGB(255, 0, 0));
		}

//		imshow("FirstImage", imageColorFirst);
//		imshow("SecondImage", imageColorSecond);

		cv::DescriptorExtractor* extractor;

		switch(this->extractor){
		case SIFT_E:
			extractor = new SiftDescriptorExtractor();
			break;
		case SURF_E:
			extractor = new SurfDescriptorExtractor();
			break;
		case BRIEF:
			extractor = new BriefDescriptorExtractor();
			break;
		case ORB_E:
			extractor = new OrbDescriptorExtractor();
			break;
		default:
			extractor = new OrbDescriptorExtractor();
		}

		Mat desciptorsImg1, desciptorsImg2;

		extractor->compute(this->firstImg, keyPointsFirst, desciptorsImg1);
		extractor->compute(this->secondImg, keyPointsSecond, desciptorsImg2);

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
			if(matches[i][0].distance < this->filter1*matches[i][1].distance && matches[i][0].distance < this->filter2*min_dist)
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

		if(good_matches.size() >= 4){
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
//		imwrite("~/ejemplowrite.png",transformed_image );

			cv::Mat result;

			warpPerspective(firstImgColor, result, H,
					cv::Size(2000, this->firstImg.rows));
			cv::Mat half(result, cv::Rect(0, 0, this->secondImg.cols, this->secondImg.rows));
			secondImgColor.copyTo(half);
			result.copyTo(secondImgColor);

			imshow("Easy Merge Result", result);

			cv::waitKey(3);
//			imwrite("~/result.png",result);
			std::swap(firstImgColor, secondImgColor);
			std::swap(firstImg, secondImg);
		}
	}

private:
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	// Propias
	float filter1;
	float filter2;
	DETECTOR detector;
	EXTRACTOR extractor;
	bool first;
	Mat firstImg;
	Mat firstImgColor;
	Mat secondImg;
	Mat secondImgColor;



};

int main(int argc, char** argv) {
	ros::init(argc, argv, "image_converter");
	PanoramCreator ic(PanoramCreator::bag, PanoramCreator::FAST, PanoramCreator::SIFT_E, 0.8, 2);
	ros::spin();
	return 0;
}
