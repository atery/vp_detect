cmake_minimum_required(VERSION 2.8)

#project name
project(vp_detect)


find_package(OpenCV 2.4.3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
 # sensor_msgs
 # std_msgs
)

catkin_package(
 # INCLUDE_DIRS include
 # CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
)

#head file path 
include_directories(
include
${catkin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

#source directory 
set(SrcFile
src/errorNIETO.cpp
src/lmmin.cpp
src/lsd.cpp
src/main.cpp
src/MSAC.cpp
)


add_executable(Vp_Detect ${SrcFile})
target_link_libraries(Vp_Detect
${OpenCV_LIBS}
${catkin_LIBRARIES}
)