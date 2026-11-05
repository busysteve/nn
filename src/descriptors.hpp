#pragma once

#include "check.hpp"
#include "cudnn_api.hpp"

namespace nn {

struct TensorDesc {
  cudnnTensorDescriptor_t raw = nullptr;
  TensorDesc() {
    check_cudnn(cudnnCreateTensorDescriptor(&raw), "cudnnCreateTensorDescriptor");
  }
  ~TensorDesc() {
    if (raw) cudnnDestroyTensorDescriptor(raw);
  }
  TensorDesc(const TensorDesc&) = delete;
  TensorDesc& operator=(const TensorDesc&) = delete;
  void set4(int n, int c, int h, int w) {
    check_cudnn(cudnnSetTensor4dDescriptor(raw, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w),
                "cudnnSetTensor4dDescriptor");
  }
};

struct FilterDesc {
  cudnnFilterDescriptor_t raw = nullptr;
  FilterDesc() { check_cudnn(cudnnCreateFilterDescriptor(&raw), "cudnnCreateFilterDescriptor"); }
  ~FilterDesc() {
    if (raw) cudnnDestroyFilterDescriptor(raw);
  }
  FilterDesc(const FilterDesc&) = delete;
  FilterDesc& operator=(const FilterDesc&) = delete;
  void set(int out_channels, int in_channels, int kernel) {
    check_cudnn(cudnnSetFilter4dDescriptor(raw, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels,
                                           in_channels, kernel, kernel),
                "cudnnSetFilter4dDescriptor");
  }
};

struct ConvDesc {
  cudnnConvolutionDescriptor_t raw = nullptr;
  ConvDesc() {
    check_cudnn(cudnnCreateConvolutionDescriptor(&raw), "cudnnCreateConvolutionDescriptor");
  }
  ~ConvDesc() {
    if (raw) cudnnDestroyConvolutionDescriptor(raw);
  }
  ConvDesc(const ConvDesc&) = delete;
  ConvDesc& operator=(const ConvDesc&) = delete;
  void set(int pad, int stride) {
    check_cudnn(cudnnSetConvolution2dDescriptor(raw, pad, pad, stride, stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor");
    check_cudnn(cudnnSetConvolutionMathType(raw, CUDNN_FMA_MATH), "cudnnSetConvolutionMathType");
    check_cudnn(cudnnSetConvolutionGroupCount(raw, 1), "cudnnSetConvolutionGroupCount");
  }
};

struct PoolDesc {
  cudnnPoolingDescriptor_t raw = nullptr;
  PoolDesc() { check_cudnn(cudnnCreatePoolingDescriptor(&raw), "cudnnCreatePoolingDescriptor"); }
  ~PoolDesc() {
    if (raw) cudnnDestroyPoolingDescriptor(raw);
  }
  PoolDesc(const PoolDesc&) = delete;
  PoolDesc& operator=(const PoolDesc&) = delete;
  void set_max(int kernel, int stride) {
    check_cudnn(cudnnSetPooling2dDescriptor(raw, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, kernel,
                                            kernel, 0, 0, stride, stride),
                "cudnnSetPooling2dDescriptor");
  }
  void set_global_avg(int height, int width) {
    check_cudnn(cudnnSetPooling2dDescriptor(raw, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                            CUDNN_NOT_PROPAGATE_NAN, height, width, 0, 0, 1, 1),
                "cudnnSetPooling2dDescriptor");
  }
};

struct ActDesc {
  cudnnActivationDescriptor_t raw = nullptr;
  ActDesc() {
    check_cudnn(cudnnCreateActivationDescriptor(&raw), "cudnnCreateActivationDescriptor");
    check_cudnn(cudnnSetActivationDescriptor(raw, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0),
                "cudnnSetActivationDescriptor");
  }
  ~ActDesc() {
    if (raw) cudnnDestroyActivationDescriptor(raw);
  }
  ActDesc(const ActDesc&) = delete;
  ActDesc& operator=(const ActDesc&) = delete;
};

}  // namespace nn
