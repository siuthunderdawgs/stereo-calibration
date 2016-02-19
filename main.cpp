/*
 * main.cpp
 *
 *  Created on: Feb 19, 2016
 *      Author: steven
 *
 *      Based on code from GitHub by Jay Rambhia (jayrambhia)
 *      See: https://github.com/jayrambhia/Vision/
 */

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>

#define WIDTH 752
#define HEIGHT 480

using namespace cv;
using namespace std;

void unsharpMask(cv::Mat& im)
{
    cv::Mat tmp;
    cv::GaussianBlur(im, tmp, cv::Size(5,5), 5);
    cv::addWeighted(im, 1.5, tmp, -0.5, 0, im);
}

int main(int argc, char* argv[])
{
    int numBoards = atoi(argv[1]);
    int board_w = atoi(argv[2]);
    int board_h = atoi(argv[3]);

    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;

    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
    }


    const char * indexfilename = "index.txt";
    std::ifstream indexfile(indexfilename);
    std::string line;
    std::vector<std::string> visualimg;
    std::vector<std::string> thermalimg;

    if(!indexfile)
    {
        std::cout << "Error: Could not open index " << indexfilename << "!" << std::endl;
        std::cout << "ABORTING!" << std::endl;
        return -1;
    }

    while(std::getline(indexfile, line))
    {
        visualimg.push_back('T' + line);
        thermalimg.push_back(line);
    }


    Mat img1, img2, gray1, gray2;
    VideoCapture cap1 = VideoCapture(1);
    VideoCapture cap2 = VideoCapture(2);

    int success = 0, k = 0;
    bool found1 = false, found2 = false;

    while (success < numBoards)
    {
        img1 = imread(visualimg.back());
        img2 = imread(thermalimg.back());
        visualimg.pop_back();
        thermalimg.pop_back();

        unsharpMask(img2);

        resize(img1, img1, Size(WIDTH, HEIGHT));
        resize(img2, img2, Size(WIDTH, HEIGHT));
        cvtColor(img1, gray1, CV_BGR2GRAY);
        cvtColor(img2, gray2, CV_BGR2GRAY);

        found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1)
        {
            cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray1, board_sz, corners1, found1);
        }

        if (found2)
        {
            cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray2, board_sz, corners2, found2);
        }

        imshow("image1", gray1);
        imshow("image2", gray2);

        k = waitKey(10);
        if (found1 && found2)
        {
            k = waitKey(0);
        }
        if (k == 27)
        {
            break;
        }
        if (k == ' ' && found1 !=0 && found2 != 0)
        {
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;

            if (success >= numBoards)
            {
                break;
            }
        }
    }

    destroyAllWindows();
    printf("Start single camera calibration\n");

    Mat CM1 = Mat(3, 3, CV_32FC1);
    Mat CM2 = Mat(3, 3, CV_32FC1);
    Mat D1;
    Mat D2;
    vector<Mat> rvecs1, tvecs1;
    vector<Mat> rvecs2, tvecs2;

    printf("Starting visual camera calibration\n");

    CM1.at<float>(0, 0) = 1;
    CM1.at<float>(1, 1) = 1;

    calibrateCamera(object_points, imagePoints1, img1.size(), CM1, D1, rvecs1, tvecs1);

    FileStorage fs1("visualcalib.yml", FileStorage::WRITE);
    fs1 << "CM1" << CM1;
    fs1 << "D1" << D1;

    printf("Visual camera calibration done\n");

    printf("Starting thermal camera calibration\n");

    CM2.at<float>(0, 0) = 1;
    CM2.at<float>(1, 1) = 1;

    calibrateCamera(object_points, imagePoints2, img2.size(), CM2, D2, rvecs2, tvecs2);

    FileStorage fs2("thermalcalib.yml", FileStorage::WRITE);
    fs2 << "CM2" << CM2;
    fs2 << "D2" << D2;

    printf("Thermal camera calibration done\n");

    Mat imgU1, imgU2;
    undistort(img1, imgU1, CM1, D1);
    undistort(img2, imgU2, CM2, D2);

    namedWindow("Undistorted Visual Image");
    imshow("Undistorted Visual Image", imgU1);
    namedWindow("Undistorted Thermal Image");
    imshow("Undistorted Thermal Image", imgU2);

    waitKey(0);
    return 0;
}
