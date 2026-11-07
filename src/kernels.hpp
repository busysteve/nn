#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nn {

void launch_add_bias(cudaStream_t stream, float* y, const float* bias, int rows, int cols);
void launch_bias_grad(cudaStream_t stream, const float* grad_y, float* grad_bias, int rows, int cols);
void launch_add2(cudaStream_t stream, float* dst, const float* a, const float* b, int64_t n);
void launch_mul(cudaStream_t stream, float* y, const float* a, const float* b, int64_t n);
void launch_gelu(cudaStream_t stream, float* y, const float* x, int64_t n);
void launch_gelu_backward(cudaStream_t stream, float* grad_x, const float* x, const float* grad_y,
                          int64_t n);
void launch_swiglu(cudaStream_t stream, float* y, const float* x, int rows, int64_t inner);
void launch_swiglu_backward(cudaStream_t stream, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner);
void launch_tanh(cudaStream_t stream, float* y, const float* x, int64_t n);
void launch_tanh_backward(cudaStream_t stream, float* grad_x, const float* y, const float* grad_y,
                          int64_t n);
void launch_gather_time(cudaStream_t stream, float* dst, const float* src, int batch, int time,
                        int features, int index);
void launch_scatter_time(cudaStream_t stream, float* dst, const float* src, int batch, int time,
                         int features, int index);
void launch_sgd(cudaStream_t stream, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum);
void launch_adamw(cudaStream_t stream, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2);
double launch_sum_squares(cudaStream_t stream, const float* x, int64_t n);
void launch_scale(cudaStream_t stream, float* x, int64_t n, float scale);

}  // namespace nn
