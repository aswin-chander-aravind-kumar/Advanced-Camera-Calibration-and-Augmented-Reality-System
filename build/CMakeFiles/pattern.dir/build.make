# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/aswinchanderaravindkumar/Desktop/Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/aswinchanderaravindkumar/Desktop/Project/build

# Include any dependencies generated for this target.
include CMakeFiles/pattern.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pattern.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pattern.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pattern.dir/flags.make

CMakeFiles/pattern.dir/src/pattern.cpp.o: CMakeFiles/pattern.dir/flags.make
CMakeFiles/pattern.dir/src/pattern.cpp.o: /Users/aswinchanderaravindkumar/Desktop/Project/src/pattern.cpp
CMakeFiles/pattern.dir/src/pattern.cpp.o: CMakeFiles/pattern.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aswinchanderaravindkumar/Desktop/Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pattern.dir/src/pattern.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pattern.dir/src/pattern.cpp.o -MF CMakeFiles/pattern.dir/src/pattern.cpp.o.d -o CMakeFiles/pattern.dir/src/pattern.cpp.o -c /Users/aswinchanderaravindkumar/Desktop/Project/src/pattern.cpp

CMakeFiles/pattern.dir/src/pattern.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pattern.dir/src/pattern.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aswinchanderaravindkumar/Desktop/Project/src/pattern.cpp > CMakeFiles/pattern.dir/src/pattern.cpp.i

CMakeFiles/pattern.dir/src/pattern.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pattern.dir/src/pattern.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aswinchanderaravindkumar/Desktop/Project/src/pattern.cpp -o CMakeFiles/pattern.dir/src/pattern.cpp.s

CMakeFiles/pattern.dir/src/functions.cpp.o: CMakeFiles/pattern.dir/flags.make
CMakeFiles/pattern.dir/src/functions.cpp.o: /Users/aswinchanderaravindkumar/Desktop/Project/src/functions.cpp
CMakeFiles/pattern.dir/src/functions.cpp.o: CMakeFiles/pattern.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aswinchanderaravindkumar/Desktop/Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pattern.dir/src/functions.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pattern.dir/src/functions.cpp.o -MF CMakeFiles/pattern.dir/src/functions.cpp.o.d -o CMakeFiles/pattern.dir/src/functions.cpp.o -c /Users/aswinchanderaravindkumar/Desktop/Project/src/functions.cpp

CMakeFiles/pattern.dir/src/functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pattern.dir/src/functions.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aswinchanderaravindkumar/Desktop/Project/src/functions.cpp > CMakeFiles/pattern.dir/src/functions.cpp.i

CMakeFiles/pattern.dir/src/functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pattern.dir/src/functions.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aswinchanderaravindkumar/Desktop/Project/src/functions.cpp -o CMakeFiles/pattern.dir/src/functions.cpp.s

CMakeFiles/pattern.dir/src/csv_util.cpp.o: CMakeFiles/pattern.dir/flags.make
CMakeFiles/pattern.dir/src/csv_util.cpp.o: /Users/aswinchanderaravindkumar/Desktop/Project/src/csv_util.cpp
CMakeFiles/pattern.dir/src/csv_util.cpp.o: CMakeFiles/pattern.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aswinchanderaravindkumar/Desktop/Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/pattern.dir/src/csv_util.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pattern.dir/src/csv_util.cpp.o -MF CMakeFiles/pattern.dir/src/csv_util.cpp.o.d -o CMakeFiles/pattern.dir/src/csv_util.cpp.o -c /Users/aswinchanderaravindkumar/Desktop/Project/src/csv_util.cpp

CMakeFiles/pattern.dir/src/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pattern.dir/src/csv_util.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aswinchanderaravindkumar/Desktop/Project/src/csv_util.cpp > CMakeFiles/pattern.dir/src/csv_util.cpp.i

CMakeFiles/pattern.dir/src/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pattern.dir/src/csv_util.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aswinchanderaravindkumar/Desktop/Project/src/csv_util.cpp -o CMakeFiles/pattern.dir/src/csv_util.cpp.s

# Object files for target pattern
pattern_OBJECTS = \
"CMakeFiles/pattern.dir/src/pattern.cpp.o" \
"CMakeFiles/pattern.dir/src/functions.cpp.o" \
"CMakeFiles/pattern.dir/src/csv_util.cpp.o"

# External object files for target pattern
pattern_EXTERNAL_OBJECTS =

pattern: CMakeFiles/pattern.dir/src/pattern.cpp.o
pattern: CMakeFiles/pattern.dir/src/functions.cpp.o
pattern: CMakeFiles/pattern.dir/src/csv_util.cpp.o
pattern: CMakeFiles/pattern.dir/build.make
pattern: /usr/local/lib/libopencv_gapi.4.9.0.dylib
pattern: /usr/local/lib/libopencv_stitching.4.9.0.dylib
pattern: /usr/local/lib/libopencv_aruco.4.9.0.dylib
pattern: /usr/local/lib/libopencv_bgsegm.4.9.0.dylib
pattern: /usr/local/lib/libopencv_bioinspired.4.9.0.dylib
pattern: /usr/local/lib/libopencv_ccalib.4.9.0.dylib
pattern: /usr/local/lib/libopencv_dnn_objdetect.4.9.0.dylib
pattern: /usr/local/lib/libopencv_dnn_superres.4.9.0.dylib
pattern: /usr/local/lib/libopencv_dpm.4.9.0.dylib
pattern: /usr/local/lib/libopencv_face.4.9.0.dylib
pattern: /usr/local/lib/libopencv_fuzzy.4.9.0.dylib
pattern: /usr/local/lib/libopencv_hfs.4.9.0.dylib
pattern: /usr/local/lib/libopencv_img_hash.4.9.0.dylib
pattern: /usr/local/lib/libopencv_intensity_transform.4.9.0.dylib
pattern: /usr/local/lib/libopencv_line_descriptor.4.9.0.dylib
pattern: /usr/local/lib/libopencv_mcc.4.9.0.dylib
pattern: /usr/local/lib/libopencv_quality.4.9.0.dylib
pattern: /usr/local/lib/libopencv_rapid.4.9.0.dylib
pattern: /usr/local/lib/libopencv_reg.4.9.0.dylib
pattern: /usr/local/lib/libopencv_rgbd.4.9.0.dylib
pattern: /usr/local/lib/libopencv_saliency.4.9.0.dylib
pattern: /usr/local/lib/libopencv_signal.4.9.0.dylib
pattern: /usr/local/lib/libopencv_stereo.4.9.0.dylib
pattern: /usr/local/lib/libopencv_structured_light.4.9.0.dylib
pattern: /usr/local/lib/libopencv_superres.4.9.0.dylib
pattern: /usr/local/lib/libopencv_surface_matching.4.9.0.dylib
pattern: /usr/local/lib/libopencv_tracking.4.9.0.dylib
pattern: /usr/local/lib/libopencv_videostab.4.9.0.dylib
pattern: /usr/local/lib/libopencv_wechat_qrcode.4.9.0.dylib
pattern: /usr/local/lib/libopencv_xfeatures2d.4.9.0.dylib
pattern: /usr/local/lib/libopencv_xobjdetect.4.9.0.dylib
pattern: /usr/local/lib/libopencv_xphoto.4.9.0.dylib
pattern: /usr/local/lib/libopencv_shape.4.9.0.dylib
pattern: /usr/local/lib/libopencv_highgui.4.9.0.dylib
pattern: /usr/local/lib/libopencv_datasets.4.9.0.dylib
pattern: /usr/local/lib/libopencv_plot.4.9.0.dylib
pattern: /usr/local/lib/libopencv_text.4.9.0.dylib
pattern: /usr/local/lib/libopencv_ml.4.9.0.dylib
pattern: /usr/local/lib/libopencv_phase_unwrapping.4.9.0.dylib
pattern: /usr/local/lib/libopencv_optflow.4.9.0.dylib
pattern: /usr/local/lib/libopencv_ximgproc.4.9.0.dylib
pattern: /usr/local/lib/libopencv_video.4.9.0.dylib
pattern: /usr/local/lib/libopencv_videoio.4.9.0.dylib
pattern: /usr/local/lib/libopencv_imgcodecs.4.9.0.dylib
pattern: /usr/local/lib/libopencv_objdetect.4.9.0.dylib
pattern: /usr/local/lib/libopencv_calib3d.4.9.0.dylib
pattern: /usr/local/lib/libopencv_dnn.4.9.0.dylib
pattern: /usr/local/lib/libopencv_features2d.4.9.0.dylib
pattern: /usr/local/lib/libopencv_flann.4.9.0.dylib
pattern: /usr/local/lib/libopencv_photo.4.9.0.dylib
pattern: /usr/local/lib/libopencv_imgproc.4.9.0.dylib
pattern: /usr/local/lib/libopencv_core.4.9.0.dylib
pattern: CMakeFiles/pattern.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/aswinchanderaravindkumar/Desktop/Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable pattern"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pattern.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pattern.dir/build: pattern
.PHONY : CMakeFiles/pattern.dir/build

CMakeFiles/pattern.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pattern.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pattern.dir/clean

CMakeFiles/pattern.dir/depend:
	cd /Users/aswinchanderaravindkumar/Desktop/Project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/aswinchanderaravindkumar/Desktop/Project /Users/aswinchanderaravindkumar/Desktop/Project /Users/aswinchanderaravindkumar/Desktop/Project/build /Users/aswinchanderaravindkumar/Desktop/Project/build /Users/aswinchanderaravindkumar/Desktop/Project/build/CMakeFiles/pattern.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/pattern.dir/depend
