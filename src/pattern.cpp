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

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <camera_index> <target> <obj_name>\n";
        return -1;
    }
    int camera_index = std::stoi(argv[1]);
    std::string target = argv[2];
    

    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video camera.\n";
        return -1;
    }
    //Resizing values for the Frame
    int resizedWidth = 640;  
    int resizedHeight = 480;
    


    while (true) {
        cv::Mat frame, gray;
        cap.read(frame);
        cv::resize(frame, frame, cv::Size(resizedWidth, resizedHeight));

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::imshow("Detected Chessboard", frame);
        int key = cv::waitKey(1);
        // Resizing the Frame
        cv::Mat dst = cv::Mat::zeros(resizedWidth, resizedHeight, CV_32FC1);
        if (key == 'q' || key == 27) { // Quit
            break;
        } else if (key == 'H' || key == 'h'){
        //harris corner detection on grayscale
        cv::cornerHarris(gray, dst, 5, 3, 0.04);

        // normalize the result
        cv::Mat dstNorm, dstNormScaled;
        cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dstNorm, dstNormScaled);

        // draw the corners
        for (int i = 0; i < dstNormScaled.rows; i++)
        {
            for (int j = 0; j < dstNormScaled.cols; j++)
            {
                if (dstNormScaled.at<uchar>(i, j) > 100)
                {
                    //drawing circles at the detected corners
                    cv::circle(frame, cv::Point(j, i), 3, cv::Scalar(0, 255, 255), 2);
                }
            }
      }
      cv::imshow("Harris Corner", frame);
    }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;

}
