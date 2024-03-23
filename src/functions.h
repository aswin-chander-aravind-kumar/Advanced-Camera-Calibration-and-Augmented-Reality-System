/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 21st March,2024

  Purpose: All the functions used in every cpp file are defined here. This makes it asy to use it without reinitialzing them each time
  
  Time travel days used : 3 days

*/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "csv_util.h"
#include <opencv2/aruco.hpp>


// Declaration of generate3DWorldPoints function
std::vector<cv::Point3f> generate3DWorldPoints(int points_per_row, int points_per_column, float square_size);

// Save the results obtained for calibration in a csv file
void saveCalibrationResultsToCSV(const cv::Mat& cameraMatrix,const cv::Mat& distCoeffs,const std::string& outputFilename);

// Loads camera matrix and distortion coefficients from a given csv file
void loadCalibrationParametersFromCSV(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, const std::string& filename);

//Draws the coordinaate axis (X,Y,Z) in a given target image based on camera's pose
void drawAxes(cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, 
              const cv::Mat& rvec, const cv::Mat& tvec, float length);


// Resize and save image function
void resizeAndSaveImage(const cv::Mat& image, const std::string& outputPath, int targetWidth);

// Used to visualize virtual chessboard above detected real world
void draw3DChessboard(cv::Mat& image, const std::vector<cv::Point2f>& corners, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Mat& rvec, const cv::Mat& tvec, int points_per_row, int points_per_column, float square_size);

// Used to draw lines between 2D points and 3D points
void drawLine(cv::Mat &img, const cv::Point2f &start, const cv::Point2f &end, const cv::Scalar &color);

void draw3DChessboardcode(cv::Mat &frame, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                          const cv::Mat &rvec, const cv::Mat &tvec,
                          int points_per_row, int points_per_column, float square_size);

// Used to see the important key features in a Function to process the custom target image and find it in a scene
cv::Mat processCustomTarget(const cv::Mat& target, const cv::Mat& scene);

// Estimates pose
bool estimatePose(const std::vector<cv::Point2f>& corners, const std::vector<cv::Point3f>& worldPoints, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Mat& rvec, cv::Mat& tvec);

// Detect chessboard corners 
bool findChessboard(cv::Mat& frame, std::vector<cv::Point2f>& corners, int points_per_row, int points_per_column);

// reads a 3D object model from an OBJ file specified by file_path and stores its vertices and faces in provided vectors
bool read_object(const std::string &file_path,
                 std::vector<cv::Point3f> &vertices,
                 std::vector<std::vector<int>> &faces,
                 float x_shift = 0.0f,
                 float y_shift = 0.0f,
                 float scale = 1.0f);

//After reading a 3D model, this function projects its vertices onto a 2D image based on the camera's
// pose (rvec and tvec), camera matrix (cam_mat), and distortion coefficients (distort_coeff)
void construct_object(const cv::Mat &rvec,
                      const cv::Mat &tvec,
                      const cv::Mat &cam_mat,
                      const cv::Mat &distort_coeff,
                      const std::vector<cv::Point3f> &vertices,
                      std::vector<std::vector<int>> &faces,
                      cv::Mat &frame);

// Load the camera calibration parameters directly
void loadCalibrationParametersDirectly(cv::Mat& cameraMatrix, cv::Mat& distCoeffs);

// Function to draw the TV model on the chessboard
bool overlayTVOnChessboard(cv::Mat& frame, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, 
                           const cv::Mat& rvec, const cv::Mat& tvec, 
                           const std::vector<cv::Point3f>& modelPoints, 
                           const std::vector<std::vector<int>>& modelFaces);

// Draw virtual cube on detected aruco markers
void drawFullCubeAruco(cv::Mat& frame, const std::vector<std::vector<cv::Point2f>>& markerCorners,const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec );

void drawFullCubeArucoMultiple(cv::Mat& frame, const std::vector<cv::Point2f>& markerCorners,const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,const cv::Mat &rvec, const cv::Mat &tvec );

// Function to draw a cube spanning all detected points on the chessboard
void drawFullCube(cv::Mat &image, const std::vector<cv::Point2f> &corners, int points_per_row, int points_per_column, float square_size);


#endif // FUNCTIONS_H

