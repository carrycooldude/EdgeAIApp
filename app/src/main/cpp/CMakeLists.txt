cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# Find Android system libraries
find_library(log-lib log)
find_library(android-lib android)

# Define your native library
add_library(edgeai_qnn SHARED 
    qnn_infer.cpp
    qnn_manager.cpp
    real_qnn_inference.cpp
)

# Link basic libraries (QNN libraries will be loaded dynamically)
target_link_libraries(edgeai_qnn
        ${log-lib}
        ${android-lib}
)
