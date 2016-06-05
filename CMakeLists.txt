cmake_minimum_required(VERSION 2.8)

#project name
project(vp_detect)


find_package(OpenCV 2.4.3 REQUIRED)

#head file path 
include_directories(
include
)

#source directory 
set(SrcFile
src/errorNIETO.cpp
src/lmmin.cpp
src/lsd.cpp
src/LSWMS.cpp
src/main.cpp
src/MSAC.cpp
)


add_executable(../bin/Vp_Detect ${SrcFile})
target_link_libraries(../bin/Vp_Detect
${OpenCV_LIBS}
)