/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 21st March,2024

  Purpose: In this program we capture video from our webcam and processes each frame to 
  detect a chessboard pattern and other patterns (  aruco , AR virtual objects), which serves as a reference for pose estimation. 
  It then refines the corners of the detected chessboard, estimates its pose relative to the camera using calibration parameters, 
  and attempts to overlay a predefined 3D model onto the chessboard in real-time. The process repeats for each 
  frame, providing an augmented reality (AR) experience by integrating virtual objects into the live video stream.

  Time travel days used : 3 days
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "functions.h"
#include "csv_util.h"
#include <opencv2/aruco.hpp>  

namespace fs = std::filesystem;

// In the command prompt we give our .exe extension , the camera Index (0 for the default camera) , 
// the target label(chessboard , aruco , AR) and some object name

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <camera_index> <target> <obj_name>\n";
        return -1;
    }

//Created a result directory in the Debug folder in order to store all the saved frames when s is pressed

    std::string resultsDirectory = "Debug/Results";
    if (!fs::exists(resultsDirectory)) {
        fs::create_directories(resultsDirectory);
    }

    int camera_index = std::stoi(argv[1]);
    std::string target = argv[2];
    std::string objName = argv[3];

    // Load target image for AR overlay if not using chessboard
    cv::Mat targetImage;
    
    // Here this was used to take in different target images for feature points extraction. The overlay code in this part didn't seem to work right
    //But it helped in determining key features when we have a target image and the image we are going to overlay the target image on
    cv::Mat targetImageForAR = cv::imread("/Users/aswinchanderaravindkumar/Desktop/Project/lenna.jpg");

    if (targetImageForAR.empty()) {
        std::cerr << "Error: Could not load target image for AR at path: " << target << std::endl;
        return -1;
    } else {
        std::cout << "Target image loaded successfully." << std::endl;
        cv::imshow("Target Image", targetImageForAR);
        cv::waitKey(1); // Display for a brief moment
    }

    // By predefined scene - Here as mentioned we are considering the enviroment on which the object is overlayed
    // To understand better what features are detected and are necessary for overlay , we initially just used 
    // classic lenna image and a picture of grandcanyon. CHose grandcanyon just to see what features is it considering imp (like the sharp edges)

    cv::Mat predefinedScene = cv::imread("/Users/aswinchanderaravindkumar/Desktop/Project/checkerboard.png");
    if (predefinedScene.empty()) {
        std::cerr << "Error: Could not load scene image." << std::endl;
        return -1;
    }else {
        std::cout << "Scene image loaded successfully." << std::endl;
        cv::imshow("Scene Image", predefinedScene);
        cv::waitKey(1); // Display for a brief moment
    }

    
    // Initialization - Defining the important params to proceed with further steps
    int points_per_row = 9, points_per_column = 6;
    cv::Size patternSize(points_per_row - 1, points_per_column - 1);
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Point3f>> point_list;
    float square_size = 1.0f;

    
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video camera.\n";
        return -1;
    }

    // Generating 3D world points from existing knowledge(properties of chessboard) in order to obtain the 2D points in 3D spacce as well
    std::vector<cv::Point3f> objp = generate3DWorldPoints(points_per_row - 1, points_per_column - 1, square_size);

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 1.0;
    cameraMatrix.at<double>(1, 1) = 1.0;
    cameraMatrix.at<double>(0, 2) = cap.get(cv::CAP_PROP_FRAME_WIDTH) / 2.0;
    cameraMatrix.at<double>(1, 2) = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 2.0;

    std::vector<double> distCoeffs(5, 0);

    int saveCount = 0;
    bool isCalibrated = false;

    while (true) {
        cv::Mat frame, gray;
        cap.read(frame);
        

        // Created the main loop suc that depending on the target name certain functions are activated
        // Used  cornerSubpix to  improve the detection of corners. Prior to this the corner detection wasn't refined
        // For target =chessboard , q - quits the code , s - saves corners and 3D points , c - calibration performed
        // Performing t - pose estimation , p - generating projection frame such that origin axis is displayed in R,G,B on the chessboard
        // V - generate a virtual chessboard . Since professor asked not to implement the general cube I instead implement virtual chessboard
        // We will map the 2 D points into 3 D space and then join those points using lines and also set the depth value such that it 
        // gives AR effect.

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if(target == "chessboard"){
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(gray, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

            if (found) {
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
                cv::drawChessboardCorners(frame, patternSize, corners, found);
            }

            cv::imshow("Detected Chessboard", frame);
            int key = cv::waitKey(1);

            if (key == 'q' || key == 27){  // Quit
                break;
            } else if (key == 's' || key == 'S') { // Save corners and 3D points
                if (found) {
                    corner_list.push_back(corners);
                    point_list.push_back(objp);
                    std::string filename = resultsDirectory + "/saved_image_" + std::to_string(saveCount++) + ".jpg";
                    resizeAndSaveImage(frame, filename, 640);
                    std::cout << "Saved image and corners for calibration\n";
                }
            } else if (key == 'c' || key == 'C') { // Calibrate
                if (corner_list.size() >= 5) {
                    std::cout << "Before Calibration:\n" << cameraMatrix << std::endl;
                    std::cout << "Initial Camera Matrix:\n" << cameraMatrix << std::endl;
                    std::cout << "Initial Distortion Coefficients:\n";
                    std::vector<cv::Mat> rvecs, tvecs;
                    double rms = cv::calibrateCamera(point_list, corner_list, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs, 0);
                    std::cout << "After Calibration:\n" << cameraMatrix << std::endl;
                    std::cout << "Calibration RMS error: " << rms << std::endl;
                    std::cout << "Calibrated Camera Matrix:\n" << cameraMatrix << std::endl;
                    std::cout << "Calibrated Distortion Coefficients:\n";
                    cv::Mat distCoeffsMat(distCoeffs.size(), 1, CV_64F, distCoeffs.data());
                    for (size_t i = 0; i < distCoeffs.size(); i++) {
                        distCoeffsMat.at<double>(i, 0) = distCoeffs[i];
                    }
                    for (int i = 0; i < distCoeffs.size(); ++i) std::cout << distCoeffs[i] << " ";
                        std::cout << std::endl;
                        isCalibrated = true;
                        saveCalibrationResultsToCSV(cameraMatrix, distCoeffsMat, resultsDirectory + "/calibration_results.csv");
                } else {
                        std::cout << "Need at least 5 images for calibration. Currently have: " << corner_list.size() << std::endl;
                }
            } else if (key == 't' || key == 'T' && isCalibrated) { // Perform pose estimation
                std::cout << "Performing rotation and translation...\n";
                cv::Mat cameraMatrix, distCoeffsMat;
                cameraMatrix = cv::Mat::eye(3, 3, CV_64F); // Assuming a 3x3 camera matrix
                loadCalibrationParametersFromCSV(cameraMatrix, distCoeffsMat, resultsDirectory + "/calibration_results.csv");
                if (found) {
                    cv::Mat rvec, tvec;
                    cv::solvePnP(objp, corners, cameraMatrix, distCoeffsMat, rvec, tvec);
                    std::cout << "Rotation Vector:\n" << rvec << std::endl;
                    std::cout << "Translation Vector:\n" << tvec << std::endl;
                } else {
                    std::cout << "No corners found for pose estimation.\n";
                }
            }else if (key == 'p' || key == 'P') {
                if (isCalibrated && found) {
                    cv::Mat rvec, tvec;
                    // Convert std::vector<double> to cv::Mat
                    cv::Mat distCoeffsMat = cv::Mat(distCoeffs).reshape(1); // Reshape to a single row if needed
                    cv::solvePnP(objp, corners, cameraMatrix, distCoeffsMat,rvec, tvec);
                    drawAxes(frame, cameraMatrix, distCoeffsMat, rvec, tvec, 2.5f); // Now passing cv::Mat
                    cv::imshow("Projection", frame);
                } else {
                    std::cout << "Chessboard not found or camera not calibrated.\n";
                }
            } else if (key == 'v' || key == 'V') {
                if (isCalibrated && found) {
                    cv::Mat cameraMatrix, distCoeffsMat;
                    loadCalibrationParametersFromCSV(cameraMatrix, distCoeffsMat, resultsDirectory + "/calibration_results.csv");

                    cv::Mat rvec, tvec;
                    bool solvedPnP = cv::solvePnP(objp, corners, cameraMatrix, distCoeffsMat, rvec, tvec);
                    if (solvedPnP) {
                        // Draw the virtual 3D chessboard
                        draw3DChessboard(frame, corners, cameraMatrix, distCoeffsMat, rvec, tvec, points_per_row, points_per_column, square_size);
                        // Display the frame with the virtual 3D chessboard
                        cv::imshow("3D Chessboard Projection", frame);
                    }
                }
            }else if (target == "AR" ) {

            // Extension part - This was an honest initial attempt for overlaying a virtual object on another scene (preferably whatever is obtained from live stream)

            cv::Mat homography = processCustomTarget(targetImageForAR, predefinedScene);

            if (!homography.empty()) {
                std::cout << "Homography found." << std::endl;
        
                std::vector<cv::Point2f> arImageCorners = {
                cv::Point2f(0, 0),
                cv::Point2f(static_cast<float>(targetImageForAR.cols), 0),
                cv::Point2f(static_cast<float>(targetImageForAR.cols), static_cast<float>(targetImageForAR.rows)),
                cv::Point2f(0, static_cast<float>(targetImageForAR.rows))
            };

                std::vector<cv::Point2f> transformedCorners;
                cv::perspectiveTransform(arImageCorners, transformedCorners, homography);

                // Create a mask for the area to overlay
                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::Point poly[4];
                for (int i = 0; i < 4; ++i) {
                    poly[i] = transformedCorners[i];
                }
                cv::fillConvexPoly(mask, poly, 4, 255);
        
                // Warp the AR image to match the perspective of the detected target in the scene
                cv::Mat warpMatrix = cv::getPerspectiveTransform(arImageCorners, transformedCorners);
                cv::Mat warpedARImage;
                cv::warpPerspective(targetImageForAR, warpedARImage, warpMatrix, frame.size());
        
                // Overlay the AR image onto the frame
                cv::Mat invertedMask;
                cv::bitwise_not(mask, invertedMask);
                cv::Mat background;
                frame.copyTo(background, invertedMask);
                cv::add(warpedARImage, background, frame, mask);

                cv::imshow("AR Overlay", frame);
            }

        
        }else{
            std::cerr << "AR overlay cannot be performed." << std::endl;
        }
    }else if (target == "aruco") {
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL));

        cv::Mat cameraMatrix, distCoeffs;
        loadCalibrationParametersFromCSV(cameraMatrix, distCoeffs, resultsDirectory + "/calibration_results.csv");

        // Create a video capture object
        cv::VideoCapture cap(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video camera.\n";
            return -1;
        }

        while (true) {
            cv::Mat frame;
            cap.read(frame);

            // Detect ArUco markers and estimate their poses
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners;
            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);
            std::cout << "Detected markers: " << markerIds.size() << std::endl;
        for (size_t i = 0; i < markerIds.size(); ++i) {
            std::cout << "Marker ID: " << markerIds[i] << std::endl;
            for (size_t j = 0; j < markerCorners[i].size(); ++j) {
                std::cout << "Corner " << j << ": " << markerCorners[i][j] << std::endl;
            }
        }

        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        if (!markerIds.empty()) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.1, cameraMatrix, distCoeffs, rvecs, tvecs);
            std::cout<<"estimated pose"<<std::endl;
            // Draw the detected markers and their axes
            
            for (size_t i = 0; i < markerIds.size(); ++i) {
                // Inside the loop where you call drawAxes
                cv::Mat rvecMat = (cv::Mat_<double>(3, 1) << rvecs[i][0], rvecs[i][1], rvecs[i][2]);
                cv::Mat tvecMat = (cv::Mat_<double>(3, 1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                std::cout<<"drawing axes"<<std::endl;
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecMat, tvecMat, 0.1);
                std::cout<<"axes drawn"<<std::endl;
                cv::putText(frame, std::to_string(markerIds[i]), markerCorners[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                drawFullCubeAruco(frame,markerCorners, cameraMatrix, distCoeffs, rvecMat, tvecMat);
            }
        }

        cv::imshow("ArUco Marker Detection", frame);
        int key = cv::waitKey(1);

        if (key == 'q' || key == 27) { // Quit
            break;
        }
    }
    }else if (target == "arucomultiple") {//for detecting mutiple arUco markers and projecting virtual objects
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL));

        cv::Mat cameraMatrix, distCoeffs;
        loadCalibrationParametersFromCSV(cameraMatrix, distCoeffs, resultsDirectory + "/calibration_results.csv");

        // Create a video capture object
        cv::VideoCapture cap(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video camera.\n";
            return -1;
        }

        while (true) {
            cv::Mat frame;
            cap.read(frame);

            // Detect ArUco markers and estimate their poses
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners;
            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);
            std::cout << "Detected markers: " << markerIds.size() << std::endl;
        for (size_t i = 0; i < markerIds.size(); ++i) {
            std::cout << "Marker ID: " << markerIds[i] << std::endl;
            for (size_t j = 0; j < markerCorners[i].size(); ++j) {
                std::cout << "Corner " << j << ": " << markerCorners[i][j] << std::endl;
            }
        }

        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        if (!markerIds.empty()) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.1, cameraMatrix, distCoeffs, rvecs, tvecs);
            std::cout<<"estimated pose"<<std::endl;
            // Draw the detected markers and their axes
            
            for (size_t i = 0; i < markerIds.size(); ++i) {
                // Inside the loop where you call drawAxes
                cv::Mat rvecMat = (cv::Mat_<double>(3, 1) << rvecs[i][0], rvecs[i][1], rvecs[i][2]);
                cv::Mat tvecMat = (cv::Mat_<double>(3, 1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                std::cout<<"drawing axes"<<std::endl;
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecMat, tvecMat, 0.1);
                std::cout<<"axes drawn"<<std::endl;
                cv::putText(frame, std::to_string(markerIds[i]), markerCorners[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                drawFullCubeArucoMultiple(frame,markerCorners[i], cameraMatrix, distCoeffs, rvecMat, tvecMat);
            }
        }

        cv::imshow("ArUco Marker Detection", frame);
        int key = cv::waitKey(1);

        if (key == 'q' || key == 27) { // Quit
            break;
        } else {
        std::cout << "Invalid target specified. Supported targets are 'chessboard' and 'armco marker'.\n";
    }


    }
    }    
    }

           
    cap.release();
    cv::destroyAllWindows();
    return 0;
}



   