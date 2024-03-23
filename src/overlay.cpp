/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 21st March,2024

  Purpose: In this program we capture video from our webcam and processes each frame to 
  detect a chessboard pattern and other patterns ( AR virtual objects), which serves as a reference for pose estimation. 
  It then refines the corners of the detected chessboard, estimates its pose relative to the camera using calibration parameters, 
  and attempts to overlay a predefined 3D model onto the chessboard in real-time. Here trying to overlay TV virtual box by apple onto  chess board

  Time travel days used : 3 days
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "functions.h"
#include <filesystem>

namespace fs = std::filesystem;

// In the command prompt we give our .exe extension , the camera Index (0 for the default camera) , and the path for our object file
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <CameraIndex> <ObjModelPath>\n";
        return -1;
    }

    int cameraIndex = std::stoi(argv[1]);
    std::string objModelPath = argv[2];

    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream from camera " << cameraIndex << "\n";
        return -1;
    }
    //Directly loading calibration parameters
    cv::Mat cameraMatrix, distCoeffs;
    loadCalibrationParametersDirectly(cameraMatrix, distCoeffs);


    //Determining the properties of the chessboard
    const int points_per_row = 9;
    const int points_per_column = 6;
    const float square_size = 0.025f; // Square size in meters

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        //Detecting chessboard corners, refining them , estimating pose and then overlaying 3D model onto the chessboard
        std::vector<cv::Point2f> corners;
        bool foundChessboard = findChessboard(frame, corners, points_per_row - 1, points_per_column - 1);

        if (foundChessboard) {
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(grayFrame, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            cv::drawChessboardCorners(frame, {points_per_row - 1, points_per_column - 1}, corners, foundChessboard);

            cv::Mat rvec, tvec;
            if (estimatePose(corners, generate3DWorldPoints(points_per_row - 1, points_per_column - 1, square_size),
                             cameraMatrix, distCoeffs, rvec, tvec)) {
                // Define the 3D points of your TV model here
                std::vector<cv::Point3f> modelPoints; //  Fill with the TV model's points
                std::vector<std::vector<int>> modelFaces; // Fill with  TV model's faces

                if (!overlayTVOnChessboard(frame, cameraMatrix, distCoeffs, rvec, tvec, modelPoints, modelFaces)) {
                    std::cerr << "Failed to overlay the 3D model.\n";
                }
            }
        }

        cv::imshow("AR Overlay", frame);
        //if (cv::waitKey(25) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
