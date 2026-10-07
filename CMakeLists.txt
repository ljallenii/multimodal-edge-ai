cmake_minimum_required(VERSION 3.20)

project(MultimodalEdgeAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(camera_capture
    src/camera_capture.cpp
)

target_link_libraries(camera_capture PRIVATE ${OpenCV_LIBS})

# ONNX Runtime is supplied externally rather than through vcpkg.
# Configure this path when generating the build directory.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set. "
        "Configure CMake with -DONNXRUNTIME_ROOT=<path-to-onnxruntime>."
    )
endif()

target_include_directories(camera_capture PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(camera_capture PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

# Copy the ONNX Runtime DLL beside the executable so Windows loads
# the same runtime version that the application was compiled against.
add_custom_command(TARGET camera_capture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:camera_capture>"
)
