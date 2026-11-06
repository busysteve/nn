#include "kernels.hpp"

#include "check.hpp"

namespace nn {
namespace {

__global__ void add_bias_kernel(float* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] += bias[i % cols];
}

__global__ void bias_grad_kernel(const float* grad_y, float* grad_bias, int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x);
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = static_cast<int>(threadIdx.x); row < rows; row += static_cast<int>(blockDim.x)) {
    sum += grad_y[static_cast<int64_t>(row) * cols + col];
  }
  __shared__ float buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) grad_bias[col] += buffer[0];
}

__global__ void add2_kernel(float* dst, const float* a, const float* b, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = a[i] + b[i];
}

__global__ void mul_kernel(float* y, const float* a, const float* b, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__device__ float gelu_forward(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + erff(x * kInvSqrt2));
}

__device__ float gelu_grad(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  constexpr float kInvSqrt2Pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + erff(x * kInvSqrt2));
  const float pdf = kInvSqrt2Pi * expf(-0.5f * x * x);
  return cdf + x * pdf;
}

__device__ float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = expf(-x);
    return 1.f / (1.f + z);
  }
  const float z = expf(x);
  return z / (1.f + z);
}

__device__ float silu(float x) { return x * sigmoid_stable(x); }

__device__ float silu_grad(float x) {
  const float s = sigmoid_stable(x);
  return s * (1.f + x * (1.f - s));
}

__global__ void gelu_kernel(float* y, const float* x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_forward(x[i]);
}

__global__ void gelu_backward_kernel(float* grad_x, const float* x, const float* grad_y, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) grad_x[i] = grad_y[i] * gelu_grad(x[i]);
}

__global__ void swiglu_kernel(float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / inner;
  const int64_t col = i - row * inner;
  const int64_t base = row * (inner * 2) + col;
  y[i] = silu(x[base]) * x[base + inner];
}

__global__ void swiglu_backward_kernel(float* grad_x, const float* x, const float* grad_y, int rows,
                                       int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / inner;
  const int64_t col = i - row * inner;
  const int64_t base = row * (inner * 2) + col;
  const float gate = x[base];
  const float value = x[base + inner];
  const float dy = grad_y[i];
  grad_x[base] = dy * value * silu_grad(gate);
  grad_x[base + inner] = dy * silu(gate);
}

__global__ void tanh_kernel(float* y, const float* x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = tanhf(x[i]);
}

__global__ void tanh_backward_kernel(float* grad_x, const float* y, const float* grad_y, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float h = y[i];
  grad_x[i] = grad_y[i] * (1.f - h * h);
}

__global__ void gather_time_kernel(float* dst, const float* src, int batch, int time, int features,
                                   int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / features;
  const int64_t col = i - row * features;
  dst[i] = src[(row * time + index) * features + col];
}

__global__ void scatter_time_kernel(float* dst, const float* src, int batch, int time, int features,
                                    int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / features;
  const int64_t col = i - row * features;
  dst[(row * time + index) * features + col] = src[i];
}

int blocks_for(int64_t n) {
  const int64_t blocks = (n + 255) / 256;
  if (blocks > 0x7fffffffLL) return 0x7fffffff;
  return static_cast<int>(blocks);
}

}  // namespace

void launch_add_bias(cudaStream_t stream, float* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  if (n <= 0) return;
  add_bias_kernel<<<blocks_for(n), 256, 0, stream>>>(y, bias, rows, cols);
  check_cuda(cudaGetLastError(), "add_bias");
}

void launch_bias_grad(cudaStream_t stream, const float* grad_y, float* grad_bias, int rows, int cols) {
  if (cols <= 0) return;
  bias_grad_kernel<<<cols, 256, 0, stream>>>(grad_y, grad_bias, rows, cols);
  check_cuda(cudaGetLastError(), "bias_grad");
}

void launch_add2(cudaStream_t stream, float* dst, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  add2_kernel<<<blocks_for(n), 256, 0, stream>>>(dst, a, b, n);
  check_cuda(cudaGetLastError(), "add2");
}

void launch_mul(cudaStream_t stream, float* y, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  mul_kernel<<<blocks_for(n), 256, 0, stream>>>(y, a, b, n);
  check_cuda(cudaGetLastError(), "mul");
}

void launch_gelu(cudaStream_t stream, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  gelu_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, n);
  check_cuda(cudaGetLastError(), "gelu");
}

void launch_gelu_backward(cudaStream_t stream, float* grad_x, const float* x, const float* grad_y,
                          int64_t n) {
  if (n <= 0) return;
  gelu_backward_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, x, grad_y, n);
  check_cuda(cudaGetLastError(), "gelu_backward");
}

void launch_swiglu(cudaStream_t stream, float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  swiglu_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, rows, inner);
  check_cuda(cudaGetLastError(), "swiglu");
}

void launch_swiglu_backward(cudaStream_t stream, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  swiglu_backward_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, x, grad_y, rows, inner);
  check_cuda(cudaGetLastError(), "swiglu_backward");
}

void launch_tanh(cudaStream_t stream, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  tanh_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, n);
  check_cuda(cudaGetLastError(), "tanh");
}

void launch_tanh_backward(cudaStream_t stream, float* grad_x, const float* y, const float* grad_y,
                          int64_t n) {
  if (n <= 0) return;
  tanh_backward_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, y, grad_y, n);
  check_cuda(cudaGetLastError(), "tanh_backward");
}

void launch_gather_time(cudaStream_t stream, float* dst, const float* src, int batch, int time,
                        int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  gather_time_kernel<<<blocks_for(n), 256, 0, stream>>>(dst, src, batch, time, features, index);
  check_cuda(cudaGetLastError(), "gather_time");
}

void launch_scatter_time(cudaStream_t stream, float* dst, const float* src, int batch, int time,
                         int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  scatter_time_kernel<<<blocks_for(n), 256, 0, stream>>>(dst, src, batch, time, features, index);
  check_cuda(cudaGetLastError(), "scatter_time");
}

}  // namespace nn
