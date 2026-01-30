cmake_minimum_required(VERSION 3.18)
project(RMT_PPAD_Deploy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86) 

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib)

include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} include)

add_executable(CoMineNet src/main.cu src/DepthAnything.cu src/JetsonDetector.cu)

target_link_libraries(CoMineNet 
    PRIVATE
    ${OpenCV_LIBS} 
    CUDA::cudart
    ${TENSORRT_LIBRARY} 
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    yaml-cpp
)
