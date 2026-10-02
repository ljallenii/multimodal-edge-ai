cmake_minimum_required(VERSION 3.20)

project(MultimodalEdgeAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(camera_capture
    src/camera_capture.cpp
)

target_link_libraries(camera_capture PRIVATE ${OpenCV_LIBS})

