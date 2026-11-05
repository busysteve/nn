#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nn {

void launch_add_bias(cudaStream_t stream, float* y, const float* bias, int rows, int cols);
void launch_bias_grad(cudaStream_t stream, const float* grad_y, float* grad_bias, int rows, int cols);
void launch_add2(cudaStream_t stream, float* dst, const float* a, const float* b, int64_t n);
void launch_sgd(cudaStream_t stream, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum);

}  // namespace nn
