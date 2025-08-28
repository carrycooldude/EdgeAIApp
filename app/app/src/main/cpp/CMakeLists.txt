cmake_minimum_required(VERSION 3.22.1)
project("qnn_infer")

# Set your QNN SDK root path
set(QNN_SDK_ROOT "C:/Users/rawat/Downloads/v2.28.0.241029/qairt/2.28.0.241029")

# Include QNN headers
include_directories(${QNN_SDK_ROOT}/include/QNN)

# Find Android system libraries
find_library(log-lib log)
find_library(android-lib android)

# Set QNN library directory
set(QNN_LIB_DIR ${QNN_SDK_ROOT}/lib/android/arm64-v8a)

# Define your native library
add_library(qnn_infer SHARED qnn_infer.cpp)

# Link everything together
target_link_libraries(qnn_infer
        ${log-lib}
        ${android-lib}
        ${QNN_LIB_DIR}/libQnnCpu.so
        ${QNN_LIB_DIR}/libQnnCommon.so
        ${QNN_LIB_DIR}/libQnnSystem.so
)