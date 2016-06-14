/*
 * Project:  vanishingPoint
 *
 * File:     main.cpp
 *
 * Contents: 
 *           
 *
 * Author:   
 *
 * Homepage: 
 */


#include <iostream>
#include <stdio.h>

#define USE_PPHT
//#define LSD_detector
#define MAX_NUM_LINES	200

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv_bridge/cv_bridge.h"
#include "ros/ros.h"
#include <image_transport/image_transport.h>

#include "MSAC.h"
#include "lsd.h"

using namespace std;
using namespace cv;


image_transport::Publisher  image_pub;

Mat mUndistMap1, mUndistMap2;//图像矫正矩阵
// Create and init MSAC
MSAC msac;



/** This function contains the actions performed for each image*/
void processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg)
{
// LSD
    vector<vector<cv::Point> > lineSegments;
    vector<cv::Point> aux;

    vector<Vec4f> lines;
    ntuple_list out;
    image_double image;
    image = new_image_double(imgGRAY.cols,imgGRAY.rows);
    //printf("imgGRAY.cols=%d,imgGRAY.rows=%d,imgGRAY.channels()=%d",imgGRAY.cols,imgGRAY.rows,imgGRAY.channels());

    for(unsigned int x=0;x<imgGRAY.rows;x++)
    {
      uchar* data_tmp=imgGRAY.ptr<uchar>(x);
      for(unsigned int y=0;y<imgGRAY.cols;y++)
          image->data[x *imgGRAY.cols + y ] = data_tmp[y];
    }
    out = lsd(image);
    for(size_t i=0;i<out->size;i++)
    {
        Vec4f line1;
        line1[0]=out->values[ i * out->dim + 0 ];
        line1[1]=out->values[ i * out->dim + 1 ];
        line1[2]=out->values[ i * out->dim + 2 ];
        line1[3]=out->values[ i * out->dim + 3 ];
        if((line1[0]-line1[2])*(line1[0]-line1[2])+(line1[1]-line1[3])*(line1[1]-line1[3]) > 2500)
        lines.push_back(line1);
    }
    for(size_t i=0; i<lines.size(); i++)
    {		
	    Point pt1, pt2;
	    pt1.x = lines[i][0];
	    pt1.y = lines[i][1];
	    pt2.x = lines[i][2];
	    pt2.y = lines[i][3];
// 	    line(outputImg, pt1, pt2, CV_RGB(0,0,0), 2);
	    /*circle(outputImg, pt1, 2, CV_RGB(255,255,255), CV_FILLED);
	    circle(outputImg, pt1, 3, CV_RGB(0,0,0),1);
	    circle(outputImg, pt2, 2, CV_RGB(255,255,255), CV_FILLED);
	    circle(outputImg, pt2, 3, CV_RGB(0,0,0),1);*/

	    // Store into vector of pairs of Points for msac
	    aux.clear();
	    aux.push_back(pt1);
	    aux.push_back(pt2);
	    lineSegments.push_back(aux);
    }

// Multiple vanishing points
    std::vector<cv::Mat> vps;			// vector of vps: vps[vpNum], with vpNum=0...numDetectedVps
    std::vector<std::vector<int> > CS;	// index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;
    
// Call msac function for multiple vanishing point estimation
    msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps); 

    
    for(int v=0; v<vps.size(); v++)
    {
	    printf("VP %d (%.3f, %.3f, %.3f)", v, vps[v].at<float>(0,0), vps[v].at<float>(1,0), vps[v].at<float>(2,0));
	    fflush(stdout);
	    double vpNorm = cv::norm(vps[v]);
	    if(fabs(vpNorm - 1) < 0.001)
	    {
		    printf("(INFINITE)");
		    fflush(stdout);
	    }
	    printf("\n");
    }
   // printf("%s","\033[1H\033[2J");
		
// Draw line segments according to their cluster
    msac.drawCS(outputImg, lineSegmentsClusters, vps);
}

void ImageSubCallback(const sensor_msgs::ImageConstPtr &msgRGB)
{  
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // Images
    cv::Mat inputImg, imgGRAY;	
    cv::Mat outputImg,outputImg_raw;
    // Other variables
    cv::Size procSize;
    int numVps = 5;
    int width = 0, height = 0;

    inputImg = cv_ptrRGB->image.clone();       
    width = inputImg.cols;
    height = inputImg.rows;
    procSize = cv::Size(width, height);

//     just for test    
//     printf("inputImg.cols=%d,inputImg.rows=%d,inputImg.channels()=%d",inputImg.cols,inputImg.rows,inputImg.channels());
//     cv:imwrite("data.jpg",inputImg);

    // Color Conversion
    if(inputImg.channels() == 3)
    {
	    cv::cvtColor(inputImg, imgGRAY, CV_BGR2GRAY);	
	    inputImg.copyTo(outputImg_raw);
    }
    else
    {
	    inputImg.copyTo(imgGRAY);
	    cv::cvtColor(inputImg, outputImg_raw, CV_GRAY2BGR);
    }

//     remap(outputImg_raw, outputImg, mUndistMap1, mUndistMap2, CV_INTER_LINEAR);
    outputImg_raw.copyTo(outputImg);//这里需要重新标定相机
    cv::cvtColor(outputImg, imgGRAY, CV_BGR2GRAY);

    processImage(msac, numVps, imgGRAY, outputImg);
    
    sensor_msgs::ImagePtr msg;
    msg = cv_bridge::CvImage(std_msgs::Header(),"rgb8",outputImg).toImageMsg();
    image_pub.publish(msg);


// View
//     imshow("Output", outputImg);
//     cv::waitKey(0);
}

void rect_image(int width,int height)
{
    cv::Mat intrinsic,distcoeffs;
    cv::FileStorage fSettings("Data/PrimeSense.yaml", cv::FileStorage::READ);

    intrinsic = Mat(3,3,CV_32F);
    intrinsic.setTo(0);
//     intrinsic.at<float>(0,0) = fSettings["Camera.fx"];
//     intrinsic.at<float>(0,2) = fSettings["Camera.cx"];
//     intrinsic.at<float>(1,1) = fSettings["Camera.fy"];
//     intrinsic.at<float>(1,2) = fSettings["Camera.cy"];
    intrinsic.at<float>(0,0) = 517.306408;
    intrinsic.at<float>(0,2) = 318.643040;
    intrinsic.at<float>(1,1) = 516.469215;
    intrinsic.at<float>(1,2) = 255.313989;
    intrinsic.at<float>(2,2) = (float)1;

    distcoeffs = Mat(4,1,CV_32F);
    distcoeffs.setTo(0);
    distcoeffs.at<float>(0) = 0.262383;
    distcoeffs.at<float>(1) = -0.953104;
    distcoeffs.at<float>(2) = -0.005358;
    distcoeffs.at<float>(3) = 0.002628;
//     distcoeffs.at<float>(0) = fSettings["Camera.k1"];
//     distcoeffs.at<float>(1) = fSettings["Camera.k2"];
//     distcoeffs.at<float>(2) = fSettings["Camera.p1"];
//     distcoeffs.at<float>(3) = fSettings["Camera.p2"];

    
    initUndistortRectifyMap(intrinsic, distcoeffs, Mat(), Mat(),cv::Size(width, height), CV_32FC1, mUndistMap1, mUndistMap2);
    
//     std::cout<<intrinsic.at<float>(0,0)<<intrinsic.at<float>(0,2)<<intrinsic.at<float>(1,1)<<std::endl;
//     std::cout<<mUndistMap1.at<float>(0)<<mUndistMap1.at<float>(1)<<mUndistMap1.at<float>(2)<<std::endl;
}

void system_initial(int width,int height)
{
  int mode = MODE_NIETO;
  bool verbose = false;
  cv::Size procSize;
  rect_image(width, height);
  procSize = cv::Size(width, height);
  msac.init(mode, procSize, verbose);
}



/** Main function*/
int main(int argc, char** argv)
{	
    ros::init(argc,argv,"Vp_detect");
    ros::start();
    
    ros::NodeHandle nh;
    system_initial(640,480);
    ros::Subscriber image_sub = nh.subscribe("/camera/rgb/image_raw", 10, ImageSubCallback); 
    image_transport::ImageTransport it(nh);
    image_pub = it.advertise("/line_segment",1);
    
    ros::spin();
    ros::shutdown();
    
    return 0;
}



