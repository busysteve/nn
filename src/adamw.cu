#include "kernels.hpp"

#include "check.hpp"

namespace nn {
namespace {

__global__ void adamw_kernel(float* param, float* m, float* v, const float* grad, int64_t n, float lr,
                             float beta1, float beta2, float eps, float weight_decay, float bc1,
                             float bc2) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = grad[i];
  const float m_new = beta1 * m[i] + (1.f - beta1) * g;
  const float v_new = beta2 * v[i] + (1.f - beta2) * g * g;
  m[i] = m_new;
  v[i] = v_new;
  const float m_hat = m_new / bc1;
  const float v_hat = v_new / bc2;
  const float update = m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i];
  param[i] = param[i] - lr * update;
}

__global__ void sum_squares_partial_kernel(const float* x, int64_t n, float* partials) {
  float sum = 0.f;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float value = x[i];
    sum += value * value;
  }
  __shared__ float buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = buffer[0];
}

__global__ void sum_partials_kernel(const float* partials, int n, float* out) {
  float sum = 0.f;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    sum += partials[i];
  }
  __shared__ float buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = buffer[0];
}

__global__ void scale_kernel(float* x, int64_t n, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= scale;
}

int grid_blocks(int64_t n) {
  const int64_t blocks64 = (n + 255) / 256;
  if (blocks64 > 0x7fffffffLL) return 0x7fffffff;
  return static_cast<int>(blocks64);
}

}  // namespace

void launch_adamw(cudaStream_t stream, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2) {
  if (n <= 0) return;
  adamw_kernel<<<grid_blocks(n), 256, 0, stream>>>(param, m, v, grad, n, lr, beta1, beta2, eps,
                                                   weight_decay, bc1, bc2);
  check_cuda(cudaGetLastError(), "adamw");
}

float launch_sum_squares(cudaStream_t stream, const float* x, int64_t n) {
  if (n <= 0) return 0.f;
  int64_t blocks64 = (n + 255) / 256;
  if (blocks64 > 1024) blocks64 = 1024;
  if (blocks64 < 1) blocks64 = 1;
  const int blocks = static_cast<int>(blocks64);
  float* scratch = nullptr;
  check_cuda(cudaMalloc(&scratch, sizeof(float) * static_cast<size_t>(blocks + 1)), "cudaMalloc");
  float* partials = scratch;
  float* out = scratch + blocks;
  try {
    sum_squares_partial_kernel<<<blocks, 256, 0, stream>>>(x, n, partials);
    check_cuda(cudaGetLastError(), "sum_squares");
    sum_partials_kernel<<<1, 256, 0, stream>>>(partials, blocks, out);
    check_cuda(cudaGetLastError(), "sum_squares");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    float host = 0.f;
    check_cuda(cudaMemcpy(&host, out, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(scratch);
    return host;
  } catch (...) {
    cudaFree(scratch);
    throw;
  }
}

void launch_scale(cudaStream_t stream, float* x, int64_t n, float scale) {
  if (n <= 0) return;
  scale_kernel<<<grid_blocks(n), 256, 0, stream>>>(x, n, scale);
  check_cuda(cudaGetLastError(), "scale");
}

}  // namespace nn
