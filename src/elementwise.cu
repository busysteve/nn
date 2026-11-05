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

}  // namespace nn
