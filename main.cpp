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
    printf("Displaying undistorted images\n");

    Mat imgU1a, imgU2a;
    undistort(img1, imgU1a, CM1, D1);
    undistort(img2, imgU2a, CM2, D2);

    namedWindow("Undistorted Visual Image");
    imshow("Undistorted Visual Image", imgU1a);
    namedWindow("Undistorted Thermal Image");
    imshow("Undistorted Thermal Image", imgU2a);

    waitKey(0);

    destroyAllWindows();
    printf("Starting stereo camera calibration\n");

    Mat R, T, E, F;

    /* BEGIN ORIGINAL INVOCATION OF STEREOCALIBRATE(...)
     * From: https://github.com/jayrambhia/Vision/blob/master/OpenCV/C%2B%2B/stereocalibrate.cpp
     *
     * stereoCalibrate(object_points, imagePoints1, imagePoints2,
     *                 CM1, D1, CM2, D2, img1.size(), R, T, E, F,
     *                 cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
     *                 CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);
     */

    /* BEGIN STACKOVERFLOW INVOCATION OF STEREOCALIBRATE(...)
     * From: http://stackoverflow.com/questions/22877869/stereocalibrate-for-different-cameras-rgb-and-infrared
     *
     * stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
     *                 CM1, D1, CM2, D2, img1.size(), R, T, E, F,
     *                 cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 50, 1e-6),
     *                 CV_CALIB_FIX_INTRINSIC + CV_CALIB_USE_INTRINSIC_GUESS);
	 */

    // MERGED INVOCATION OF STEREOCALIBRATE(...)
    stereoCalibrate(object_points, imagePoints1, imagePoints2,
            CM1, D1, CM2, D2, img1.size(), R, T, E, F,
            cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
            CV_CALIB_FIX_INTRINSIC + CV_CALIB_USE_INTRINSIC_GUESS);

    FileStorage fs3("stereocalib.yml", FileStorage::WRITE);
    fs3 << "CM1" << CM1;
    fs3 << "CM2" << CM2;
    fs3 << "D1" << D1;
    fs3 << "D2" << D2;
    fs3 << "R" << R;
    fs3 << "T" << T;
    fs3 << "E" << E;
    fs3 << "F" << F;

    printf("Stereo camera calibration done\n");
    printf("Starting stereo camera rectification\n");

    Mat R1, R2, P1, P2, Q;

    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

    fs3 << "R1" << R1;
    fs3 << "R2" << R2;
    fs3 << "P1" << P1;
    fs3 << "P2" << P2;
    fs3 << "Q" << Q;

    printf("Stereo camera rectification done\n");
    printf("Applying Undistort\n");

    Mat map1x, map1y, map2x, map2y;

    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

    printf("Undistort done\n");
    printf("Displaying remapped images\n");

    Mat imgU1b, imgU2b;
    remap(img1, imgU1b, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(img2, imgU2b, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

    namedWindow("Remapped Visual Image");
    imshow("Remapped Visual Image", imgU1b);
    namedWindow("Remapped Thermal Image");
    imshow("Remapped Thermal Image", imgU2b);

    waitKey(0);

    return 0;
}
