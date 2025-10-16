cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# 16 KB page size compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# Samsung S25 Ultra compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_SAMSUNG_S25_ULTRA_COMPAT")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_SAMSUNG_S25_ULTRA_COMPAT")

# ExecuTorch flags for v79 (based on official Qualcomm AI HUB Apps)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QNN=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QUALCOMM=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_CONTEXT_BINARIES_V79=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_SOC_MODEL_69=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_HEXAGON_V79=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ARCHITECTURE_AARCH64_ANDROID=1")

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
    improved_llama_inference.cpp
    real_executorch_qnn.cpp # Real ExecuTorch + QNN integration
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
