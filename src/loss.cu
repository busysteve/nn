#include "nn/loss.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "nn/error.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <utility>

namespace nn {
namespace {

__global__ void softmax_ce_kernel(const float* logits, const int64_t* labels, float* grad,
                                  float* per_sample, int n, int classes, int* bad) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= n) return;
  const float* row_logits = logits + static_cast<int64_t>(row) * classes;
  float* row_grad = grad + static_cast<int64_t>(row) * classes;

  float thread_max = -1e30f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, row_logits[c]);
  }
  __shared__ float scratch[128];
  scratch[threadIdx.x] = thread_max;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float max_logit = scratch[0];
  __syncthreads();

  float thread_sum = 0.f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_sum += expf(row_logits[c] - max_logit);
  }
  scratch[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
    __syncthreads();
  }
  const float sum = scratch[0];
  const int64_t label = labels[row];
  if (label < 0 || label >= classes) {
    if (threadIdx.x == 0) atomicExch(bad, 1);
    return;
  }
  const float inv_sum = 1.f / sum;
  const float inv_n = 1.f / static_cast<float>(n);
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    const float softmax = expf(row_logits[c] - max_logit) * inv_sum;
    const float indicator = (static_cast<int64_t>(c) == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (threadIdx.x == 0) per_sample[row] = -(row_logits[label] - (max_logit + logf(sum)));
}

}  // namespace

DeviceIndices::DeviceIndices(Context& ctx, int64_t n) : ctx_(&ctx), n_(n) {
  if (n <= 0) throw Error("empty labels");
  check_cuda(cudaMalloc(&data_, static_cast<size_t>(n) * sizeof(int64_t)), "cudaMalloc");
}

DeviceIndices::~DeviceIndices() {
  if (!data_) return;
  if (ctx_) cudaStreamSynchronize(stream_of(*ctx_));
  cudaFree(data_);
  data_ = nullptr;
}

DeviceIndices::DeviceIndices(DeviceIndices&& other) noexcept
    : ctx_(other.ctx_), data_(other.data_), n_(other.n_) {
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
}

DeviceIndices& DeviceIndices::operator=(DeviceIndices&& other) noexcept {
  if (this == &other) return *this;
  if (data_) {
    if (ctx_) cudaStreamSynchronize(stream_of(*ctx_));
    cudaFree(data_);
  }
  ctx_ = other.ctx_;
  data_ = other.data_;
  n_ = other.n_;
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
  return *this;
}

void DeviceIndices::copy_from_host(const int64_t* src, int64_t n) {
  if (n != n_) throw Error("label count mismatch");
  check_cuda(cudaMemcpyAsync(data_, src, static_cast<size_t>(n) * sizeof(int64_t),
                             cudaMemcpyHostToDevice, stream_of(*ctx_)),
             "cudaMemcpyAsync");
}

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels) {
  if (logits.rank() != 2) throw Error("logits rank");
  const int n = logits.shape()[0];
  const int classes = logits.shape()[1];
  if (n <= 0 || classes <= 1) throw Error("logits shape");
  if (labels.size() != n) throw Error("label count mismatch");

  LossResult result{0.f, Tensor(ctx, {n, classes})};
  float* per_sample = nullptr;
  int* bad = nullptr;
  check_cuda(cudaMalloc(&per_sample, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc");
  check_cuda(cudaMalloc(&bad, sizeof(int)), "cudaMalloc");
  cudaStream_t stream = stream_of(ctx);
  check_cuda(cudaMemsetAsync(bad, 0, sizeof(int), stream), "cudaMemsetAsync");
  softmax_ce_kernel<<<n, 128, 0, stream>>>(logits.data(), labels.data(), result.grad_logits.data(),
                                           per_sample, n, classes, bad);
  check_cuda(cudaGetLastError(), "softmax_cross_entropy");

  thrust::device_ptr<float> begin(per_sample);
  const float sum = thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n, 0.f);
  ctx.synchronize();
  int bad_host = 0;
  check_cuda(cudaMemcpy(&bad_host, bad, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaFree(per_sample);
  cudaFree(bad);
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n);
  return result;
}

}  // namespace nn
