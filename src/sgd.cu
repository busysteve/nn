#include "kernels.hpp"

#include "check.hpp"

namespace nn {
namespace {

__global__ void sgd_kernel(float* param, float* velocity, const float* grad, int64_t n, float lr,
                           float momentum) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float updated = momentum * velocity[i] + grad[i];
  velocity[i] = updated;
  param[i] = param[i] - lr * updated;
}

}  // namespace

void launch_sgd(cudaStream_t stream, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum) {
  if (n <= 0) return;
  const int64_t blocks64 = (n + 255) / 256;
  const int blocks = blocks64 > 0x7fffffffLL ? 0x7fffffff : static_cast<int>(blocks64);
  sgd_kernel<<<blocks, 256, 0, stream>>>(param, velocity, grad, n, learning_rate, momentum);
  check_cuda(cudaGetLastError(), "sgd");
}

}  // namespace nn
