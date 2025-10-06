cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# 16 KB page size compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# Samsung S25 Ultra compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_SAMSUNG_S25_ULTRA_COMPAT")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_SAMSUNG_S25_ULTRA_COMPAT")

# Set alignment for 16 KB page size
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-z,max-page-size=16384")

# Find Android system libraries
find_library(log-lib log)
find_library(android-lib android)

# Define your native library
add_library(edgeai_qnn SHARED 
    qnn_infer.cpp
    qnn_manager.cpp
    real_qnn_inference.cpp
)

# Set properties for 16 KB page size compatibility
set_target_properties(edgeai_qnn PROPERTIES
    LINK_FLAGS "-Wl,-z,max-page-size=16384"
    POSITION_INDEPENDENT_CODE ON
)

# Link basic libraries (QNN libraries will be loaded dynamically)
target_link_libraries(edgeai_qnn
        ${log-lib}
        ${android-lib}
)
