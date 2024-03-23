# Advanced-Camera-Calibration-and-Augmented-Reality-System

In this program we capture video from our webcam and processes each frame to detect a chessboard pattern and other patterns (  aruco , AR virtual objects), which serves as a reference for pose estimation. It then refines the corners of the detected chessboard, estimates its pose relative to the camera using calibration parameters, and attempts to overlay a predefined 3D model onto the chessboard in real-time. The process repeats for each frame, providing an augmented reality (AR) experience by integrating virtual objects into the live video stream.


## Tasks

- **Detect and Extract Target Corners**: Implement a system for detecting a target and extracting target corners using functions like `findChessboardCorners`, `cornerSubPix`, and `drawChessboardCorners`.

- **Select Calibration Images**: Allow users to specify calibration images and save corner locations along with corresponding 3D world points when the key 's' or 'S' is pressed.

- **Calibrate the Camera**: Use `cv::calibrateCamera` to generate camera calibration parameters, including the camera matrix, distortion coefficients when the key 'c' or 'C' is pressed .

- **Calculate Current Position of the Camera**: Estimate the camera's pose using `solvePNP` and visualize rotation and translation data when the key 't' or 'T' is pressed .

- **Project Outside Corners or 3D Axes**: Project 3D points onto the image plane in real-time and display 3D axes on the target when the key 'p' or 'P' is pressed .

- **Create a Virtual Object**: Construct a virtual object in 3D world space and project it onto the image, ensuring the object maintains orientation with camera movements when the key 'v' or 'V' is pressed.

- **Detect Robust Features**: The user needs to run the pattern.cpp where the program implement feature detection (e.g., Harris corners) and visualize feature points in the image stream when the key 'h' or 'H' is pressed .
- **ArUco Markers**: When the target input by the is aruco, then the program would detect the aruco markers, estimates the pose, draws axes on the detected aruco markers and place a virtual object on the aruco markers.
-  **Multiple ArUco Markers**: When the target input by the is arucomultiple, then the program would detect the multiple aruco markers, estimates the pose, draws axes on the detected aruco markers and place a virtual object on the aruco markers simultaneously.

## Dependencies

- Make sure the openCV_contrib folder has the files related to Aruco
  
## Build Instructions

1. Clone the repository.
2. Install OpenCV if not already installed.
3. Build the project using c make.
4. Run the executable file.

## Usage

- Follow the instructions provided in each task to execute specific functionalities.
- Ensure that the system has access to a video stream or pre-recorded video files for processing.

## Documentation

Refer to the project's source code for detailed comments and documentation. Additional information can be found in the project report.


## Skills Obtained

- C++ Proficiency
- Computer Vision
- OpenCV
- OpenGL
- Camera Calibration
- Feature Detection
- Augmented Reality
- 3D Projection
- Image Processing
- Algorithm Implementation
- Feature Extraction
- Pose Estimation
- Data Analysis


