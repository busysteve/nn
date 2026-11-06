cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT}
  PATH_SUFFIXES include)
find_library(CUDNN_LIBRARY cudnn
  HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT}
  PATH_SUFFIXES lib lib64)
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR
    "cuDNN not found. Set CUDNN_INCLUDE_DIR and CUDNN_LIBRARY.")
endif()

add_library(nn STATIC
  src/context.cpp
  src/error.cpp
  src/tensor.cpp
  src/init.cpp
  src/config.cpp
  src/elementwise.cu
  src/linear.cpp
  src/relu.cpp
  src/gelu.cpp
  src/swiglu.cpp
  src/layer_norm.cu
  src/dropout.cpp
  src/rnn.cpp
  src/last_step.cpp
  src/conv2d.cpp
  src/max_pool2d.cpp
  src/batch_norm2d.cpp
  src/flatten.cpp
  src/global_avg_pool2d.cpp
  src/residual_block.cpp
  src/network.cpp
  src/optim.cpp
  src/sgd.cu
  src/loss.cu
  src/checkpoint.cpp
  src/builders.cpp
)

target_include_directories(nn
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDNN_INCLUDE_DIR})

target_link_libraries(nn PUBLIC CUDA::cudart CUDA::cublas PRIVATE ${CUDNN_LIBRARY})

set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(nn PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)

set_target_properties(nn PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON)

add_executable(nn_tests
  tests/test_main.cpp
  tests/test_shapes.cpp
  tests/test_gradients.cpp
  tests/test_checkpoint.cpp
  tests/test_mnist.cpp
  tests/mnist.cpp
)

target_link_libraries(nn_tests PRIVATE nn)
target_include_directories(nn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

enable_testing()
add_test(NAME shapes COMMAND nn_tests shapes)
add_test(NAME grads COMMAND nn_tests grads)
add_test(NAME ckpt COMMAND nn_tests ckpt)
add_test(NAME mnist COMMAND nn_tests mnist)
set_tests_properties(shapes grads ckpt PROPERTIES TIMEOUT 900)
set_tests_properties(mnist PROPERTIES TIMEOUT 7200)
