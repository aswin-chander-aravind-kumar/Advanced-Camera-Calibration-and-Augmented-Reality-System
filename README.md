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

- Ensure you have CMake version 3.15 or higher installed on your system
- Create a CMake lists that ensures that all the files are linked together accurately 
- Download all the cpp and .h files to your local machine
- Place the cpp and .h files in src folder. Creating a results folder in build/Debug filter
- Saving all the videos in the results Directory
- Now create a build folder and use cmake .. to build it
- To run your executable enter the debug folder and run the ./main executable along with the required arguments
- Additionally linked csv_util.cpp(professor's), csv_util.h(professor's), overlay.cpp and pattern.cpp to implement additional tasks
-Saving features.csv file in build/Debug/results/calibrations_results.csv

## Instructions to implement the tasks

Instructions for implementing different tasks:

**In main.cpp**:

Chessboard Detection and Calibration

- q or `sc: Quit the program.
- s: Save detected corners and 3D points for calibration.
- c: Perform camera calibration using detected corners and saved images.
- t: Perform pose estimation for the detected chessboard.
- p: Generate projection frame with origin axis displayed in R, G, B on the chessboard.
- v: Generate a virtual chessboard by mapping 2D points to 3D space and overlaying virtual objects.

**AR Overlay**
- AR target:
  - Overlay a virtual object onto the detected scene.


**ArUco Marker Detection**
- aruco target:
  - Detect ArUco markers and estimate their poses.
  - Draw detected markers and their axes on the video frame.
  
**Multiple ArUco Marker Detection**
- arucomultiple target:
  - Detect multiple ArUco markers and estimate their poses.
  - Draw detected markers and their axes on the video frame.

**In pattern.cpp:**

**Harris Corner Detection**
- `H` or `h`: Detect corners using the Harris corner detection algorithm.
- Overlay the detected corners on the original video frame.


**Please see the command prompts code for main.cpp, pattern.cpp and overlay.cpp to implement respective commands**

main.cpp - std::cout << "Usage: " << argv[0] << " <camera_index> <target> <obj_name>\n"

pattern.cpp - std::cout << "Usage: " << argv[0] << " <camera_index> <target> <obj_name>\n"

overlay.cpp - std::cerr << "Usage: " << argv[0] << " <CameraIndex> <ObjModelPath>\n"

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


