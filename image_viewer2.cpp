#include <pcl/io/openni_grabber.h>			//librarie pour l'acquisition des images

#include <pcl/visualization/cloud_viewer.h>		//librarie pour analyser les images récupérées

#include <pcl/visualization/image_viewer.h>

#include <pcl/impl/point_types.hpp>

#include <pcl/point_cloud.h>

#include <opencv2/opencv.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;


class ImageVIewer{
public:
  
  pcl::visualization::CloudViewer viewer;
  VideoWriter record;
  Mat frame2;
  int it;
  pcl::PointCloud<pcl::PointXYZRGBA> nuage;
  pcl::PointCloud<pcl::PointXYZRGBA> nuage2;
  Mat depth;
  

  ImageVIewer() : viewer ("viewer") {}

   void cloud_cb_(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud){		//fonction <> =>classe template 
   
    if(!viewer.wasStopped()){
    
      /*for(int i=0;i<cloud->width;i++){
    for (int j=0;j<cloud->height;j++){ 
      if (i>300 && j>300)
      std::cout <<cloud->width<< cloud->height <<std::endl;
    }}
    */
      nuage=*cloud;
      //viewer.showCloud(nuage);//on montre le viewer tant qu'on ne l'a pas arreté  
    }    
  }

  
  void image_cb_ (const boost::shared_ptr<openni_wrapper::Image>& img)
{
  
    Mat frame= getFrame (img); 
    
    //imshow( "Display Image", frame);
        
    if (it>10){
      frame2=frame;
      it=0;
    }else{
      it++;
    }
    
    sift_demo(frame2,frame);
    //frame=cornerHarris_demo(frame);
    //frame=surf_demo(frame);    
    //frame=fast_demo(frame);    
    
    imshow( "Harris Image", frame);
    
    //record.write(frame); //I used a cv::VideoWriter vw because I needed to get a Video from the robot point of view
    waitKey(1);
}


Mat cornerHarris_demo( Mat dst ){
  Mat dst_norm, dst_norm_scaled;
  /// Detector parameters
  int thresh = 200;
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  /// Detecting corners
  cornerHarris(dst, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  
  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );
  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ ){ 
    for( int i = 0; i < dst_norm.cols; i++ ){
      if( (int) dst_norm.at<float>(j,i) > thresh ){
        circle( dst_norm_scaled, Point( i, j ), 5, Scalar(0), 2, 8, 0 );
      }
    }
  }
  return dst_norm_scaled;
}

void sift_demo( Mat dst,Mat dst2 ){
 
  SurfFeatureDetector detector (1500); 

  std::vector<KeyPoint> keypoints_1,keypoints_2;  
  detector.detect(dst, keypoints_1);
  detector.detect(dst2, keypoints_2);
  //drawKeypoints(dst, keypoints_1, dst);
  
  SurfDescriptorExtractor extractor;
  Mat descriptors_1, descriptors_2;
  extractor.compute( dst, keypoints_1, descriptors_1 );
  extractor.compute( dst2, keypoints_2, descriptors_2 );
  
  //-- Step 3: Matching descriptor vectors with a brute force matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
  pcl::PointXYZRGBA point;
    
  double max_dist = 0; double min_dist = 100;

    //filtrage des associations ratées
  for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  std::vector< DMatch > good_matches;
  for( int i = 0; i < descriptors_1.rows; i++ ){
    if( matches[i].distance < 3*min_dist ){    
    good_matches.push_back( matches[i]);    
    }
  }
  nuage2.clear();
  //nuage2=nuage;
 for(int i=0;i<dst.cols;i++){
    for (int j=0;j<dst.rows;j++){ 
  
  //for(int i=0;i<matches.size;i++){  
   
      point.x=i;
      point.y=j;

       point.g=nuage.at(i,j).g;//dst.at<cv::Vec3b>(i,j)[1];
      point.b=nuage.at(i,j).b;//dst.at<cv::Vec3b>(i,j)[2];
      point.r=nuage.at(i,j).r;
	//if (depth.at<float>(i,j)==depth.at<float>(i,j)){
	  if (nuage.at(i,j).z==nuage.at(i,j).z){
//	std::cout<<depth.at<float>(i,j)<<endl;
	 point.z=nuage.at(i,j).z*100;//depth.at<float>(j,i)*100;
	  
	}
	nuage2.push_back(point);
      
    }
  }
 
  //-- Draw matches
  Mat img_matches;
  drawMatches( dst, keypoints_1, dst2, keypoints_2, good_matches, img_matches );  
  imshow("Matches", img_matches );
   
}

Mat surf_demo( Mat dst ){
 
  //cv::SurfAdjuster detector(detect);
  cv::FeatureDetector * detector = new cv::SURF(200.0);
  
  std::vector<KeyPoint> keypoints;
  
  detector->detect(dst, keypoints);
  drawKeypoints(dst, keypoints, dst);
  return dst;
}

/*
Mat fast_demo( Mat dst ){
 
  //cv::SurfAdjuster detector(detect);
  cv::FeatureDetector * detector = new cv::FAST(200);
  
  std::vector<KeyPoint> keypoints;
  
  detector->detect(dst, keypoints);
  drawKeypoints(dst, keypoints, dst);
  return dst;
}
*/

void depth_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& img)
{
    Mat depth2=Mat(img->getHeight(),img->getWidth(),DataType<float>::type);
    
    img->fillDepthImage(depth.cols,depth.rows,(float*)depth.data,depth.step);
    //depth.convertTo(depth2,CV_32FC1,0.125/2,0);
    normalize( depth, depth2, 0, 255, CV_MINMAX, CV_64FC1, Mat() );
    //convertScaleAbs( depth, depth );
    imshow( "Depth Image", depth2);
    //std::cout << depth<< std::endl;

    waitKey(1);
}

 Mat getFrame (const boost::shared_ptr<openni_wrapper::Image> &img)
{
  Mat frameRGB=Mat(img->getHeight(),img->getWidth(),CV_8UC3);  
  img->fillRGB(frameRGB.cols,frameRGB.rows,frameRGB.data,frameRGB.step);
  Mat frameBGR;
  cvtColor(frameRGB,frameBGR,CV_BGR2GRAY);//CV_RGB2BGR);

  return frameBGR;
}


  
  void run(){
    
    depth=Mat(480,640,DataType<float>::type);
    
   pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr nuage3(&nuage2);// (new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::PointXYZRGBA point;
   
   it=1000;
   pcl::OpenNIGrabber* interface =new pcl::OpenNIGrabber();//creation d'un objet interface qui vient de l'include openni_grabber
   //namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
   namedWindow( "Harris Image", CV_WINDOW_AUTOSIZE );
   //namedWindow( "Depth Image", CV_WINDOW_AUTOSIZE );
  // VideoCapture capture(1);
  // Mat frame;
  // capture >> frame;
  // record=VideoWriter("/home/guerric/Bureau/test.avi", CV_FOURCC('M','J','P','G'), 30, frame.size(), true);
   
   boost::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> 
  f = boost::bind (&ImageVIewer::cloud_cb_, this, _1);

  boost::function<void(const boost::shared_ptr<openni_wrapper::Image>&)> 
  g = boost::bind (&ImageVIewer::image_cb_, this, _1);
  
  boost::function<void(const boost::shared_ptr<openni_wrapper::DepthImage>&)> 
  h = boost::bind (&ImageVIewer::depth_cb_, this, _1);


  interface->registerCallback (f);
  interface->registerCallback (g);
  interface->registerCallback (h);
  
   
   interface->start();
   //on reste dans cet état d'acquisition tant qu'on ne stoppe pas dans le viewer

   
   while(!viewer.wasStopped()){
     boost::this_thread::sleep(boost::posix_time::seconds(1));	//met la fonction en attente pendant une seconde <=> sleep(1) mais plus précis pour les multicores     
     viewer.showCloud(nuage3);     
  }
   
  interface->stop();
  record.release();
  destroyAllWindows();  

  }
  void pcloud();
  
};

int main() {
  ImageVIewer kinect;
  kinect.run();

return 0;
}