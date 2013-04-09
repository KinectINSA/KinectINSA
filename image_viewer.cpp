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



using namespace cv;


class ImageVIewer{
public:
  
  pcl::visualization::CloudViewer viewer;
  VideoWriter record;
  
  ImageVIewer() : viewer ("viewer") {}

   void cloud_cb_(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud){		//fonction <> =>classe template  
  
    if(!viewer.wasStopped()){
      viewer.showCloud(cloud);//on montre le viewer tant qu'on ne l'a pas arreté
    }    
  }

  
  void image_cb_ (const boost::shared_ptr<openni_wrapper::Image>& img)
{
    Mat frame= getFrame (img);   
    imshow( "Display Image", frame);
    
    frame=cornerHarris_demo(frame);
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




void depth_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& img)
{
    Mat depth=Mat(img->getHeight(),img->getWidth(),CV_64FC1);
    Mat depth2=Mat(img->getHeight(),img->getWidth(),CV_64FC1);
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
   pcl::OpenNIGrabber* interface =new pcl::OpenNIGrabber();//creation d'un objet interface qui vient de l'include openni_grabber
   namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
   namedWindow( "Harris Image", CV_WINDOW_AUTOSIZE );
   namedWindow( "Depth Image", CV_WINDOW_AUTOSIZE );
   VideoCapture capture(1);
   Mat frame;
   capture >> frame;
   record=VideoWriter("/home/guerric/Bureau/test.avi", CV_FOURCC('M','J','P','G'), 30, frame.size(), true);
   
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
     boost::this_thread::sleep(boost::posix_time::seconds(50));	//met la fonction en attente pendant une seconde <=> sleep(1) mais plus précis pour les multicores
   }
   
  interface->stop();
  record.release();
  destroyAllWindows();
  

  }
  
};

int main() {
  ImageVIewer kinect;
  kinect.run();

return 0;
}