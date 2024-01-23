cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(Detections)

set(CMAKE_PREFIX_PATH "C:/Users/developer/Downloads/libtorch-win-shared-with-deps-2.1.2/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)

set(OpenCV_DIR "C:/Users/developer/Downloads/opencv/build/x64/vc14/lib/")
find_package(OpenCV REQUIRED)

set(SAMPLE_SOURCES main.cpp)
set(TARGET_NAME sampleSAM)

set(TORCH_LIB "C:/Users/developer/Downloads/libtorch-win-shared-with-deps-2.1.2/libtorch/lib/*.lib")
set(NVINFER_LIBS "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib/nvinfer.lib")
set(NVINFER_LIBS_PARSER "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib/nvonnxparser.lib")

include(CMakePrintHelpers)
cmake_print_variables(TORCH_LIBRARIES)

cmake_print_variables(TORCH_INCLUDE_DIRS)

# commons
include_directories("./commons")

include_directories(${TARGET_NAME} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include/")
include_directories(${TARGET_NAME} "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include/")

add_executable(${TARGET_NAME} ${SAMPLE_SOURCES})
target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} ${NVINFER_LIBS})
target_link_libraries(${TARGET_NAME} ${NVINFER_LIBS_PARSER})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
