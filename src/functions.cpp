/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 21st March,2024

  Purpose: This program implements essential functionalities for augmented reality (AR) applications, 
  including the detection and refinement of chessboard pattern for camera pose estimation, overlaying 3D models 
  onto detected patterns, and handling camera calibration data. 
  These functions are used in both main.cpp and overlay.cpp. They work in background . They contain detailed implementation of various functions
  
  Time travel days used : 3 days

*/


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "functions.h" 
#include "csv_util.h"
#include <filesystem>
#include <fstream>

// Implementation of generate3DWorldPoints
// Assumes point_set is a vector of Vec3f, where each Vec3f is a 3D point (x, y, z)
// Here Z coordinate is always assumed to be zero 
std::vector<cv::Point3f> generate3DWorldPoints(int points_per_row, int points_per_column, float square_size) {
    std::vector<cv::Point3f> worldPoints;
    for (int i = 0; i < points_per_column; ++i) {
        for (int j = 0; j < points_per_row; ++j) {
            worldPoints.push_back(cv::Point3f(j * square_size, i * square_size, 0.0f));
        }
    }
    return worldPoints;
}


// Save the results obtained for calibration in a csv file
void saveCalibrationResultsToCSV(
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const std::string& outputFilename) {
    
    std::vector<float> cameraMatrixData;
    std::vector<float> distCoeffsData;
    
    // Flatten camera matrix to a vector
    for (int i = 0; i < cameraMatrix.rows; ++i) {
        for (int j = 0; j < cameraMatrix.cols; ++j) {
            cameraMatrixData.push_back(static_cast<float>(cameraMatrix.at<double>(i, j)));
        }
    }
    
    // Flatten distortion coefficients to a vector
    for (int i = 0; i < distCoeffs.rows; ++i) {
        for (int j = 0; j < distCoeffs.cols; ++j) {
            distCoeffsData.push_back(static_cast<float>(distCoeffs.at<double>(i, j)));
        }
    }
    
    // Appending these values - helps in reusing them avoiding recalibration 
    // "CameraMatrix" - identifier for the camera matrix row in the CSV
    append_image_data_csv(const_cast<char*>(outputFilename.c_str()), const_cast<char*>("CameraMatrix"), cameraMatrixData, 1);
    
    // "DistCoeffs" - identifier for the distortion coefficients row in the CSV
    // Using professors csv_util.cpp to call upon the function append_image_data_csv
    append_image_data_csv(const_cast<char*>(outputFilename.c_str()), const_cast<char*>("DistCoeffs"), distCoeffsData, 0);
}


// Loads camera matrix and distortion coefficients from a given csv file
void loadCalibrationParametersFromCSV(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open calibration file: " << filename << std::endl;
        return;
    }

    // Initializing matrices to zeros before filling them with loaded data
    // Helps in correctly sizing and formatting for use in camera calibration

    cameraMatrix = cv::Mat::zeros(3, 3, CV_64F); // Initialize cameraMatrix with zeros
    distCoeffs = cv::Mat::zeros(1, 5, CV_64F); // Initialize distCoeffs with zeros

    std::string line;
    std::getline(file, line); // Read the CameraMatrix line
    if (!line.empty()) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); // Skip the label "CameraMatrix"
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (!std::getline(ss, cell, ',')) {
                    std::cerr << "Error: Invalid Camera Matrix format." << std::endl;
                    return;
                }
                cameraMatrix.at<double>(i, j) = std::stod(cell);
            }
        }
    }

    std::getline(file, line); // Read the DistCoeffs line
    if (!line.empty()) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); // Skip the label "DistCoeffs"
        for (int i = 0; i < 5; ++i) {
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Error: Invalid Distortion Coefficients format." << std::endl;
                return;
            }
            distCoeffs.at<double>(0, i) = std::stod(cell);
        }
    }

    file.close();
}

// Draws the coordinaate axis (X,Y,Z) in a given target image based on camera's pose 
void drawAxes(cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, 
              const cv::Mat& rvec, const cv::Mat& tvec, float length) {
    // Define points in 3D space for axes
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(length, 0, 0)); // X axis
    axisPoints.push_back(cv::Point3f(0, length, 0)); // Y axis
    axisPoints.push_back(cv::Point3f(0, 0, length)); // Z axis

    // Project these 3D points onto 2 D image plane using the predefined function
    // We consider camera's rotation (rvec),translation (tvec),intrinsic parameters(cameraMatrix),distortion coeffiecients (distCoeffs)
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw axes - (red - X , green - Y ,blue -Z)
    // The axes drawn directly onto the detected corners target image
    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2); // X in red
    cv::line(image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2); // Y in green
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2); // Z in blue
}

// Resize and save image function
// Did this to increase the computation time. Initially there was extreme lag in the processing
// After using this it improved significantly
// While maintaining the aspect ratio , the target image is resized and saved
void resizeAndSaveImage(const cv::Mat& image, const std::string& outputPath, int targetWidth = 640) {
    cv::Mat resizedImage;
    double aspectRatio = image.cols / static_cast<double>(image.rows);
    int resizedHeight = static_cast<int>(targetWidth / aspectRatio);
    
    // Resize image
    cv::resize(image, resizedImage, cv::Size(targetWidth, resizedHeight));

    // Save resized image
    cv::imwrite(outputPath, resizedImage);
}


//Used to visualize virtual chessboard above detected real world
//chessboard , creating an illusion of depth such that it seems like 
//the virtual chessboard is floating above the detected chessboard
void draw3DChessboard(cv::Mat& image, const std::vector<cv::Point2f>& corners, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Mat& rvec, const cv::Mat& tvec, int points_per_row, int points_per_column, float square_size) {
    // The height where the virtual chessboard will appear above the actual chessboard
    // This is used to simulate the floating effect
    float virtual_board_height = square_size * 2; 

    std::vector<cv::Point3f> virtualBoardCorners;
    // Generate 3D points for the chessboard corners in its local coordinate system
    //Project 3D points into 2D image plane using cameras pose and intrinsic parameters

    for (int i = 0; i < points_per_column; ++i) {
        for (int j = 0; j < points_per_row; ++j) {
            virtualBoardCorners.push_back(cv::Point3f(j * square_size, i * square_size, -virtual_board_height));
        }
    }

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(virtualBoardCorners, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Drawing lines between the projected points to form virtual chessboard grid overlaying it onto the image
    // Draw the chessboard grid based on the projected points
    for (int i = 0; i < points_per_column - 1; ++i) {
        for (int j = 0; j < points_per_row - 1; ++j) {
            int idx = i * points_per_row + j;
            // Draw lines to the right and down from each point
            if (j < points_per_row - 1) {
                cv::line(image, projectedPoints[idx], projectedPoints[idx + 1], cv::Scalar(0, 255, 0), 2);
            }
            if (i < points_per_column - 1) {
                cv::line(image, projectedPoints[idx], projectedPoints[idx + points_per_row], cv::Scalar(255, 0, 0), 2);
            }
        }
    }
}

void draw3DChessboardcode(cv::Mat &frame, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                          const cv::Mat &rvec, const cv::Mat &tvec,
                          int points_per_row, int points_per_column, float square_size) {
    // Define the 3D points of the chessboard corners in the chessboard's reference frame
    std::vector<cv::Point3f> chessboardCorners3D;
    for (int i = 0; i < points_per_column; ++i) {
        for (int j = 0; j < points_per_row; ++j) {
            chessboardCorners3D.push_back(cv::Point3f(j * square_size, i * square_size, 0));
        }
    }

    // Project the 3D points back to the image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(chessboardCorners3D, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw the chessboard grid
    for (int i = 0; i < points_per_column; ++i) {
        for (int j = 0; j < points_per_row; ++j) {
            int idx = i * points_per_row + j;
            // Skip the last row and column points
            if (j < points_per_row - 1)
                cv::line(frame, imagePoints[idx], imagePoints[idx + 1], cv::Scalar(0, 255, 0), 2);
            if (i < points_per_column - 1)
                cv::line(frame, imagePoints[idx], imagePoints[idx + points_per_row], cv::Scalar(255, 0, 0), 2);
        }
    }
}


// Function to process the custom target image and find it in a scene
// Returns a homography matrix that maps points from the target image to the scene
cv::Mat processCustomTarget(const cv::Mat& target, const cv::Mat& scene) {
    // Initialize ORB detector
    //cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

    // ORB detector - wasn't working as expected hence used akaze instead
    // for detecting features / keypoints in target image and previously mentioned scene(environment on wich target image is overlayed)
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

    // Detect and compute keypoints and descriptors for both images
    std::vector<cv::KeyPoint> keypointsTarget, keypointsScene;
    cv::Mat descriptorsTarget, descriptorsScene;


        
    akaze->detectAndCompute(target, cv::noArray(), keypointsTarget, descriptorsTarget);
    akaze->detectAndCompute(scene, cv::noArray(), keypointsScene, descriptorsScene);

    std::cout << "Keypoints in Target: " << keypointsTarget.size() << std::endl;
    std::cout << "Keypoints in Scene: " << keypointsScene.size() << std::endl;

    // Check if keypoints are detected
    if (keypointsTarget.empty() || keypointsScene.empty()) {
        std::cerr << "Error: No keypoints found in one or both images." << std::endl;
        return cv::Mat(); // Return an empty matrix
    }

    // Debug: Visualize keypoints
    cv::Mat keypointImageTarget, keypointImageScene;
    cv::Mat targetImageForAR = cv::imread("C:/Users/shrut/OneDrive/Desktop/PROJECT41/lenna.jpg");
    //cv::Mat scene = cv::imread("C:/Users/shrut/OneDrive/Desktop/PROJECT41/img.jpg");
    cv::waitKey(0); 
    
    cv::drawKeypoints(targetImageForAR, keypointsTarget, keypointImageTarget);
    cv::drawKeypoints(scene, keypointsScene, keypointImageScene);
    cv::imshow("Keypoints Target", keypointImageTarget);
    cv::imshow("Keypoints Scene", keypointImageScene);
    cv::waitKey(0); // Wait for a key press

    // Match descriptors using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsTarget, descriptorsScene, matches);
    std::cout << "Matches found: " << matches.size() << std::endl;
    if (descriptorsTarget.empty() || descriptorsScene.empty()) {
        std::cerr << "Error: One or both descriptors are empty." << std::endl;
        return cv::Mat();
         // Or handle the error as needed
    }

    if (descriptorsTarget.type() != descriptorsScene.type() ||
        descriptorsTarget.cols != descriptorsScene.cols) {
        std::cerr << "Error: Descriptors are not compatible for matching." << std::endl;
        return cv::Mat();
 // Or handle the error as needed
    }

    // Check if matches are found
    if (matches.empty()) {
        std::cerr << "Error: No matches found between descriptors." << std::endl;
        return cv::Mat(); // Or handle the error as needed
    }



    // Sort matches by score
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    // Select the top matches
    const int numGoodMatches = matches.size() * 0.15;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Extract location of good matches
    std::vector<cv::Point2f> pointsTarget, pointsScene;
    for(const auto& match : matches) {
        pointsTarget.push_back(keypointsTarget[match.queryIdx].pt);
        pointsScene.push_back(keypointsScene[match.trainIdx].pt);
    }

    // Find homography
    cv::Mat homography = cv::findHomography(pointsTarget, pointsScene, cv::RANSAC);
    // Check if a valid homography was found
    if (homography.empty()) {
        std::cerr << "Error: Could not find a valid homography." << std::endl;
        return cv::Mat(); // Or handle the error as needed
    }

    return homography;
}


// Functions specifically for overlay.cpp


bool findChessboard(cv::Mat& frame, std::vector<cv::Point2f>& corners, int points_per_row, int points_per_column) {
    // Assuming the chessboard pattern is points_per_row x points_per_column
    cv::Size patternSize(points_per_row, points_per_column);
    bool found = cv::findChessboardCorners(frame, patternSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // Improve corner accuracy
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }

    return found;
}


//reads a 3D object model from an OBJ file specified by file_path and stores its vertices and faces in provided vectors.
bool read_object(const std::string &file_path, std::vector<cv::Point3f> &vertices, std::vector<std::vector<int>> &faces, float x_shift, float y_shift, float scale) {
    std::ifstream objFile(file_path);
    if (!objFile.is_open()) {
        return false; // Return false if the file couldn't be opened
    }

    //Opens the OBJ file and reads it line by line, distinguishing vertex (v) and face (f)
    //For vertices, it scales and shifts them according to provided parameters (x_shift, y_shift, scale), allowing for adjustments in the object's position and size when rendered.
    //For faces, it stores indices of vertices that form each face
    //Returns true if the file is successfully read and false if any errors occurs
    std::string line;
    while (getline(objFile, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            if (!(ss >> x >> y >> z)) {
                objFile.close();
                return false; // Return false if the vertex format is incorrect
            }
            vertices.push_back(cv::Point3f(x * scale + x_shift, y * scale + y_shift, z * scale));
        } else if (prefix == "f") {
            std::vector<int> face;
            int vertexIndex;
            char slash;
            while (ss >> vertexIndex) {
                if (ss.fail()) {
                    objFile.close();
                    return false; // Return false if the face format is incorrect
                }
                face.push_back(vertexIndex);
                // Skip over texture/normal indices if present
                ss.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
            }
            if (!face.empty()) {
                faces.push_back(face);
            } else {
                objFile.close();
                return false; // Return false if no face data is present
            }
        }
    }

    objFile.close();
    return true; // Return true if everything is read successfully
}


//After reading a 3D model, this function projects its vertices onto a 2D image based on the camera's
// pose (rvec and tvec), camera matrix (cam_mat), and distortion coefficients (distort_coeff)
//Uses cv::projectPoints to transform the 3D vertices into 2D points on the image
void construct_object(const cv::Mat &rvec,
                      const cv::Mat &tvec,
                      const cv::Mat &cam_mat,
                      const cv::Mat &distort_coeff,
                      const std::vector<cv::Point3f> &vertices,
                      std::vector<std::vector<int>> &faces,
                      cv::Mat &frame) {
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(vertices, rvec, tvec, cam_mat, distort_coeff, projectedPoints);
//Drawing lines between these projected points according to the model's face definitions, effectively reconstructing the 3D model's geometry in the 2D image space
    for (const auto &face : faces) {
        for (size_t i = 0; i < face.size(); ++i) {
            // Connect each point to the next, and the last point back to the first
            cv::line(frame,
                     projectedPoints[face[i] - 1], // OBJ indices are 1-based
                     projectedPoints[face[(i + 1) % face.size()] - 1],
                     cv::Scalar(0, 255, 0), // Color
                     2); // Thickness
        }
    }
}



bool estimatePose(const std::vector<cv::Point2f>& corners, const std::vector<cv::Point3f>& worldPoints, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Mat& rvec, cv::Mat& tvec) {
    // SolvePnP to find the rotation and translation vectors
    return cv::solvePnP(worldPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);
}

// Load the camera calibration parameters directly
void loadCalibrationParametersDirectly(cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    // Camera matrix
    cameraMatrix = (cv::Mat_<double>(3, 3) << 523.0867, 0.0, 335.5621,
                                              0.0, 530.1334, 245.2716,
                                              0.0, 0.0, 1.0);

    // Distortion coefficients
    distCoeffs = (cv::Mat_<double>(1, 5) << 0.3270, -1.4929, 0.0032, 0.0036, 1.4914);
}


// Here I have a TV virtual box model from apple
// I converted the Usdz file into obj file to use the coordinates
// Function to draw the TV model on the chessboard
bool overlayTVOnChessboard(cv::Mat& frame, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, 
                           const cv::Mat& rvec, const cv::Mat& tvec, 
                           const std::vector<cv::Point3f>& modelPoints, 
                           const std::vector<std::vector<int>>& modelFaces) {
    // Project the 3D model onto the 2D image
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(modelPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw the projected model
    for (const auto& face : modelFaces) {
        for (size_t i = 0; i < face.size(); ++i) {
            // Draw lines connecting the model's vertices
            cv::line(frame, projectedPoints[face[i] - 1], projectedPoints[face[(i + 1) % face.size()] - 1], cv::Scalar(0, 255, 0), 2);
        }
    }
    return true;
}


// Function to draw a cube spanning all detected points on the chessboard
void drawFullCube(cv::Mat &image, const std::vector<cv::Point2f> &corners, int points_per_row, int points_per_column, float square_size) {
    // Ensure there are enough corners
    if (corners.size() != points_per_row * points_per_column) {
        std::cerr << "Not enough corners to draw a full cube." << std::endl;
        return;
    }

    // Define cube depth
    float depth = square_size * 3; // Adjust the depth as needed

    // Get the four corners of the chessboard
    cv::Point2f bottomLeft = corners.front(); // Bottom-left corner
    cv::Point2f bottomRight = corners[points_per_row - 1]; // Bottom-right corner
    cv::Point2f topLeft = corners[(points_per_column - 1) * points_per_row]; // Top-left corner
    cv::Point2f topRight = corners.back(); // Top-right corner

    // Calculate the top square of the cube based on the bottom corners and depth
    std::vector<cv::Point2f> topSquare(4);
    topSquare[0] = bottomLeft + cv::Point2f(-depth, -depth);
    topSquare[1] = bottomRight + cv::Point2f(depth, -depth);
    topSquare[2] = topRight + cv::Point2f(depth, depth);
    topSquare[3] = topLeft + cv::Point2f(-depth, depth);

    // Draw bottom square of the cube
    cv::line(image, bottomLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
    cv::line(image, bottomRight, topRight, cv::Scalar(0, 0, 255), 2);
    cv::line(image, topRight, topLeft, cv::Scalar(0, 0, 255), 2);
    cv::line(image, topLeft, bottomLeft, cv::Scalar(0, 0, 255), 2);

    // Draw top square of the cube
    cv::line(image, topSquare[0], topSquare[1], cv::Scalar(0, 255, 0), 2);
    cv::line(image, topSquare[1], topSquare[2], cv::Scalar(0, 255, 0), 2);
    cv::line(image, topSquare[2], topSquare[3], cv::Scalar(0, 255, 0), 2);
    cv::line(image, topSquare[3], topSquare[0], cv::Scalar(0, 255, 0), 2);

    // Draw vertical lines of the cube
    cv::line(image, bottomLeft, topSquare[0], cv::Scalar(255, 0, 0), 2);
    cv::line(image, bottomRight, topSquare[1], cv::Scalar(255, 0, 0), 2);
    cv::line(image, topRight, topSquare[2], cv::Scalar(255, 0, 0), 2);
    cv::line(image, topLeft, topSquare[3], cv::Scalar(255, 0, 0), 2);
}

//Draw 3D virtual cube on detected aruco markers
void drawFullCubeAruco(cv::Mat& frame, const std::vector<std::vector<cv::Point2f>>& markerCorners,const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec ) {
    float cubeSize = 0.1f; // Adjust this value as needed

    // Define the 3D points of the cube
    std::vector<cv::Point3f> cubePoints;
    cubePoints.push_back(cv::Point3f(0, 0, 0));
    cubePoints.push_back(cv::Point3f(cubeSize, 0, 0));
    cubePoints.push_back(cv::Point3f(cubeSize, cubeSize, 0));
    cubePoints.push_back(cv::Point3f(0, cubeSize, 0));
    cubePoints.push_back(cv::Point3f(0, 0, -cubeSize));
    cubePoints.push_back(cv::Point3f(cubeSize, 0, -cubeSize));
    cubePoints.push_back(cv::Point3f(cubeSize, cubeSize, -cubeSize));
    cubePoints.push_back(cv::Point3f(0, cubeSize, -cubeSize));

    // Draw the cube for each detected marker
    for (size_t i = 0; i < markerCorners.size(); ++i) {
        // Project the 3D cube points onto the image plane
        std::vector<cv::Point2f> cubePoints2D;
        cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, cubePoints2D);

        // Draw the cube edges
        for (int j = 0; j < 4; ++j) {
            cv::line(frame, cubePoints2D[j], cubePoints2D[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cubePoints2D[j + 4], cubePoints2D[((j + 1) % 4) + 4], cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cubePoints2D[j], cubePoints2D[j + 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}
//Draw 3D virtual cube on multiple detected aruco markers
void drawFullCubeArucoMultiple(cv::Mat& frame, const std::vector<cv::Point2f>& markerCorners,const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec ) {
    float cubeSize = 0.1f; // Adjust this value as needed

    // Define the 3D points of the cube
    std::vector<cv::Point3f> cubePoints;
    cubePoints.push_back(cv::Point3f(0, 0, 0));
    cubePoints.push_back(cv::Point3f(cubeSize, 0, 0));
    cubePoints.push_back(cv::Point3f(cubeSize, cubeSize, 0));
    cubePoints.push_back(cv::Point3f(0, cubeSize, 0));
    cubePoints.push_back(cv::Point3f(0, 0, -cubeSize));
    cubePoints.push_back(cv::Point3f(cubeSize, 0, -cubeSize));
    cubePoints.push_back(cv::Point3f(cubeSize, cubeSize, -cubeSize));
    cubePoints.push_back(cv::Point3f(0, cubeSize, -cubeSize));

    // Draw the cube for each detected marker
    for (size_t i = 0; i < markerCorners.size(); ++i) {
        // Project the 3D cube points onto the image plane
        std::vector<cv::Point2f> cubePoints2D;
        cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, cubePoints2D);

        // Draw the cube edges
        for (int j = 0; j < 4; ++j) {
            cv::line(frame, cubePoints2D[j], cubePoints2D[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cubePoints2D[j + 4], cubePoints2D[((j + 1) % 4) + 4], cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cubePoints2D[j], cubePoints2D[j + 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}

