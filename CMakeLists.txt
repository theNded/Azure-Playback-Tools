cmake_minimum_required(VERSION 3.0.0)

project(k4arecorder)
find_package(k4a REQUIRED)
find_package(k4arecord REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(libjpeg-turbo)
include_directories(libjpeg-turbo)

add_executable(main main.cpp)
target_link_libraries(main k4a::k4a k4a::k4arecord turbojpeg ${OpenCV_LIBS})
