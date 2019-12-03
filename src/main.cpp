/*
 *  point_compute.cpp
 *
 *  Created by tangguojun  on 2019/4/1.
 *
 */
//#define OPENCV2_VERSION "opencv2"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h>  

#ifdef OPENCV2_VERSION
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#else
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include <vector>
#endif

using namespace cv;
using namespace std;
using namespace pcl;

static void print_help() {
    cout << "\nDemo stereo matching converting L and R images into disparity and point clouds\n";
    cout << "\nUsage: stereo_match <left_image> <right_image>\n";
}

//空洞填充
void insertDepth32f(Mat& depth) {
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i) {
        int id1 = i * width;
        for (int j = 0; j < width; ++j) {
            int id2 = id1 + j;
            if (data[id2] > 1e-3) {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i) {
        int id1 = i * width;
        for (int j = 1; j < width; ++j) {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i) {
        int id1 = i * width;
        for (int j = 0; j < width; ++j) {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1) {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i) {
            int id1 = i * width;
            for (int j = 0; j < width; ++j) {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0) {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201) {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

#ifdef OPENCV2_VERSION
//opencv2 的GC视差图计算方法
void gc_match(string &limg_filename, string &rimg_filename, Mat *dispmat) {
    IplImage * img1 = cvLoadImage(limg_filename.c_str(), 0);
    IplImage * img2 = cvLoadImage(rimg_filename.c_str(), 0);
    CvStereoGCState* GCState=cvCreateStereoGCState(64, 3);
    assert(GCState);
    cout << "start matching using GC" << endl;
    CvMat* gcdispleft=cvCreateMat(img1->height, img1->width, CV_16S);
    CvMat* gcdispright=cvCreateMat(img2->height, img2->width, CV_16S);
    CvMat* gcvdisp=cvCreateMat(img1->height, img1->width, CV_8U);
    int64 t = getTickCount();
    cvFindStereoCorrespondenceGC(img1, img2, gcdispleft, gcdispright, GCState);
    t = getTickCount() - t;
    cout << "Time elapsed:" << t*1000/getTickFrequency() << endl;

    cvNormalize(gcdispleft, gcvdisp, 0, 255, CV_MINMAX);
    cvSaveImage("GC_left_disparity.png", gcvdisp);
    //cvNamedWindow("GC_disparity",0);
    //cvShowImage("GC_disparity",gcvdisp);
    //cvWaitKey(0);
    *dispmat = Mat(gcvdisp, true);
    cvReleaseMat(&gcdispleft);
    cvReleaseMat(&gcdispright);
    cvReleaseMat(&gcvdisp);
}
#else
//opencv3 SGBM视差图计算方法
bool sgbm_match(string &limg_filename, string &rimg_filename, Mat *dispmat) {
    int color_mode = -1;
    Mat limg = imread(limg_filename, color_mode);
    Mat rimg = imread(rimg_filename, color_mode);
    if (limg.empty()) {
        cout << "Command-line parameter error: could not load the first input image file\n";
        return false;
    }
    if (rimg.empty()) {
        cout << "Command-line parameter error: could not load the second input image file\n";
        return false;
    }

    Mat disp;
    Size img_size = limg.size();
    int numberOfDisparities = ((img_size.width / 8) + 15) & -16;
    int SADWindowSize = 0;
    int cn = limg.channels();
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    sgbm->setBlockSize(sgbmWinSize);
    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    sgbm->compute(limg, rimg, *dispmat);

    insertDepth32f(*dispmat);
    //disp.convertTo(*dispmat, CV_8U);
    imwrite("./disparity.png", *dispmat);
    //waitKey();

    return true;
}
#endif

void viewerOneOff(visualization::PCLVisualizer& viewer) {
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
}

//视差图点云计算
void disparity_point_compute(float fx, float fy, float cx, float cy, float tx,
                    float doffs, Mat &dispmat, string &limg_filename) {
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    Mat limg = imread(limg_filename, -1);
    int rowNumber = limg.rows;
    int colNumber = limg.cols;

    for (int u = 0; u < rowNumber; ++u) {
        for (int v = 0; v < colNumber; ++v) {
            ushort d = dispmat.ptr<ushort>(u)[v];
			if (d == 0) {
            	continue;
            }
			PointXYZRGB p;
			
			// depth			
			p.z = fx * tx / (d + doffs);
			p.x = (v - cx) * p.z / fx;
			p.y = (u - cy) * p.z / fy;
			
			p.y = -p.y;
			p.z = -p.z;
            //std::cout << "---------->z:" << p.z << ", x:" << p.x << ", y:" << p.y << std::endl;
			// RGB
			p.b = limg.ptr<uchar>(u)[v * 3];
			p.g = limg.ptr<uchar>(u)[v * 3 + 1];
			p.r = limg.ptr<uchar>(u)[v * 3 + 2];

			cloud->points.push_back(p);
        }
    }
    std::cout << "====>height:" << dispmat.rows << ", width:" << dispmat.cols << std::endl;
    cloud->height = dispmat.rows;
	cloud->width = dispmat.cols;
	cloud->points.resize(cloud->height * cloud->width);

    io::savePCDFileASCII("point_ascii.pcd", *cloud);

    visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    viewer.runOnVisualizationThreadOnce(viewerOneOff);
    while (!viewer.wasStopped())
    {
        //cout << "------------------->" << endl;
    }
}

//深度图点云计算
double depth_compute_distance(float fx, float fy, float cx, float cy, 
                    Mat &depth, int obj_x1, int obj_y1, int obj_x2, int obj_y2) {
    double distances = 0.0;
    int count = 0;

    for (int u = obj_x1; u < obj_x2; ++u) {
        for (int v = obj_y1; v < obj_y2; ++v) {
            float d = depth.at<uchar>(v, u);
			if (d == 0) {
            	continue;
            }
			PointXYZRGB p;
			
			// depth			
			p.z = d;
			p.x = (v - cx) * p.z / fx;
			p.y = (u - cy) * p.z / fy;
			
			p.y = -p.y;
			p.z = -p.z;
            
			double distance = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
            //std::cout << distance << "---->z:" << p.z << ", x:" << p.x << ", y:" << p.y << std::endl;
            distances += distance;
            count++;
        }
    }

    return distances / count;
}

// string split
vector<string> split(const string& str, const string& delim)
{  
	vector<string> res;  
	if("" == str) return res;  
	//先将要切割的字符串从string类型转换为char*类型  
	char *strs = new char[str.length() + 1] ;  
	strcpy(strs, str.c_str());   
 
	char *d = new char[delim.length() + 1];  
	strcpy(d, delim.c_str());  
 
	char *p = strtok(strs, d);  
	while(p) 
  {  
		string s = p; //分割得到的字符串转换为string类型  
		res.push_back(s); //存入结果数组  
		p = strtok(NULL, d);  
	}  
 
	return res;  
}

//深度图点云计算
void depth_point_compute(float fx, float fy, float cx, float cy, 
                    Mat &depth, string &limg_filename) {
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    Mat limg = imread(limg_filename, -1);
    vector<string> res = split(limg_filename, "/.");
    cout << "test " << res[3] << endl;
    string pcd_name = "./data/point_cloud/pcd_2/" + res[3] + ".pcd";
    int rowNumber = limg.rows;
    int colNumber = limg.cols;

    for (int u = 0; u < rowNumber; ++u) {
        for (int v = 0; v < colNumber; ++v) {
            ushort d = depth.ptr<ushort>(u)[v];
			if (d == 0) {
            	continue;
            }
			PointXYZRGB p;
			
			// depth			
			p.z = d;
			p.x = (v - cx) * p.z / fx;
			p.y = (u - cy) * p.z / fy;
			
			p.y = -p.y;
			p.z = -p.z;
            double dis = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
            // std::cout << dis << "---------->z:" << p.z << ", x:" << p.x << ", y:" << p.y << std::endl;
			// RGB
			p.b = limg.ptr<uchar>(u)[v * 3];
			p.g = limg.ptr<uchar>(u)[v * 3 + 1];
			p.r = limg.ptr<uchar>(u)[v * 3 + 2];

			cloud->points.push_back(p);
        }
    }
    std::cout << "====>height:" << depth.rows << ", width:" << depth.cols << std::endl;
    cloud->height = depth.rows;
	cloud->width = depth.cols;
	cloud->points.resize(cloud->height * cloud->width);

    io::savePCDFileASCII(pcd_name, *cloud);

    // visualization::CloudViewer viewer("Cloud Viewer");
    // viewer.showCloud(cloud);
    // viewer.runOnVisualizationThreadOnce(viewerOneOff);
    // while (!viewer.wasStopped())
    // {
    //     //cout << "------------------->" << endl;
    // }
}

int main(int argc, char** argv)
{
    std::string limg_filename = "";
    std::string rimg_filename = "";
    std::string dimg_filename = "";
    std::string mode = "";
    std::string intrinsic_filename = "";
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    #ifdef OPENCV2_VERSION
    cv::CommandLineParser parser(argc, argv,
        "{l||}{r||}{d||}{m||}{i||}{x1||}{y1||}{x2||}{y2||}");
    limg_filename = parser.get<std::string>("l");
    rimg_filename = parser.get<std::string>("r");
    dimg_filename = parser.get<std::string>("d");
    mode = parser.get<std::string>("m");
    intrinsic_filename = parser.get<std::string>("i");
    x1 = atoi(parser.get<std::string>("x1").c_str());
    y1 = atoi(parser.get<std::string>("y1").c_str());
    x2 = atoi(parser.get<std::string>("x2").c_str());
    y2 = atoi(parser.get<std::string>("y2").c_str());
    #else
    cv::CommandLineParser parser(argc, argv,
        "{l||}{r||}{d||}{m||}{i||}{x1||}{y1||}{x2||}{y2||}{help h||}");
    if(parser.has("help")) {
        print_help();
        return -1;
    }
    limg_filename = parser.get<std::string>("l");
    rimg_filename = parser.get<std::string>("r");
    dimg_filename = parser.get<std::string>("d");
    mode = parser.get<std::string>("m");
    intrinsic_filename = parser.get<std::string>("i");
    x1 = atoi(parser.get<std::string>("x1").c_str());
    y1 = atoi(parser.get<std::string>("y1").c_str());
    x2 = atoi(parser.get<std::string>("x2").c_str());
    y2 = atoi(parser.get<std::string>("y2").c_str());
    #endif

    if (mode == "lr") {
        if(limg_filename.empty() || rimg_filename.empty() || intrinsic_filename.empty()){
            cout << "Command-line parameter error: both left and right images must be specified\n";
            return -1;
        }
    } else if (mode == "dp") {
        if(limg_filename.empty() || dimg_filename.empty() || intrinsic_filename.empty()){
            cout << "Command-line parameter error: both left and right images must be specified\n";
            return -1;
        }
    } else if (mode == "di") {
        if((x2 <= x1) || (y2 <= y1) || dimg_filename.empty() || intrinsic_filename.empty()){
            cout << "Command-line parameter error: both left and right images must be specified\n";
            return -1;
        }
    } else {
        cout << "mode:---->" << mode << " error!\n";
        return -1;
    }
    // reading intrinsic parameters
    Mat k, x;
    cout << "----->" << intrinsic_filename << endl;
    FileStorage fs(intrinsic_filename.c_str(), FileStorage::READ);
    if(!fs.isOpened())
    {
        cout << "Failed to open file " << intrinsic_filename.c_str() << endl;
        return -1;
    }
    fs["K"] >> k;
    fs["X"] >> x;
    float fx = (float)k.at<double>(0, 0);
    float fy = (float)k.at<double>(1, 1);
    float cx = (float)k.at<double>(0, 2);
    float cy = (float)k.at<double>(1, 2);
    float tx = (float)x.at<double>(0, 0) * 1000;
    float doffs = (float)x.at<double>(0, 1);
    cout << "fx:" << fx << ", fy:" << fy << endl;
    cout << "cx:" << cx << ", cy:" << cy << endl;
    cout << "tx:" << tx << ", doffs:" << doffs << endl;
    //point compute
    if (mode == "lr") {
        //get disp
        Mat disp;
        #ifdef OPENCV2_VERSION
        gc_match(limg_filename, rimg_filename, &disp);
        #else
        sgbm_match(limg_filename, rimg_filename, &disp);
        #endif

        disparity_point_compute(fx, fy, cx, cy, tx, doffs, disp, limg_filename);
    } else if (mode == "dp") {
        Mat depth = imread(dimg_filename, -1);

        depth_point_compute(fx, fy, cx, cy, depth, limg_filename);
    } else if (mode == "di") {
        Mat depth = imread(dimg_filename, -1);
        float distance = depth_compute_distance(fx, fy, cx, cy, depth, x1, y1, x2, y2);
        cout << "distance: " << distance << "m.\n";
    }

    return 0;
}