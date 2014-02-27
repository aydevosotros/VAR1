#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


static const std::string OPENCV_WINDOW = "Image window";

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

public:
  ImageConverter() : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
      &ImageConverter::imageCbSift, this);
//    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCbSift(const sensor_msgs::ImageConstPtr& msg){
	  cv_bridge::CvImagePtr cv_ptr;
	  try
	  {
		  cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
	  }
	  catch (cv_bridge::Exception& e)
	  {
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
	  }
	  vector<KeyPoint> keyPoints;
	  Mat mask;
	  Mat imageColor;
	  Mat src_gray;
	  cvtColor(src_gray, imageColor, CV_GRAY2BGR);

	  cv::FeatureDetector* detector;
	  detector = new SiftFeatureDetector(
		 0, // nFeatures
		 4, // nOctaveLayers
		 0.04, // contrastThreshold
		 10, //edgeThreshold
		 1.6 //sigma
		 );

	  detector->detect(src_gray, keyPoints, mask);

	  for (size_t i = 0; i < keyPoints.size(); i++) {
		  circle(imageColor, keyPoints[i].pt, 3, CV_RGB(255, 0, 0));
	  }
	  Mat output;

//	  drawKeypoints(img, keypoints, output, Scalar::all(-1));

	  namedWindow("meh", CV_WINDOW_AUTOSIZE);
	  imshow("meh", output);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
	  cv_bridge::CvImagePtr cv_ptr;
	      try
	      {
	        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
	      }
	      catch (cv_bridge::Exception& e)
	      {
	        ROS_ERROR("cv_bridge exception: %s", e.what());
	        return;
	      }
	      std::cerr<<" imagecb: "<<msg->header.frame_id<<" : "<<msg->header.seq<<" : "<<msg->header.stamp<<std::endl;

	      cv::Mat src_gray;

	   	  cvtColor( cv_ptr->image, src_gray, CV_BGR2GRAY );

	   	  imshow("BN-Image", src_gray);
	     Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");
	     std::vector<cv::KeyPoint> points;
	     detector->detect(src_gray, points);


	      cv::Mat imageColor;
	      cvtColor(src_gray, imageColor, CV_GRAY2BGR);

	      for (size_t i = 0; i < points.size(); i++)
	      {
	        circle(imageColor, points[i].pt, 3, CV_RGB(255, 0, 0));
	      }
	      //se pueden pintar tambien con esta funcion
	      //drawKeypoints(imageColor, points, imageColor, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

	      imshow("Fast keypoints", imageColor);


	      cv::waitKey(3);

  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
