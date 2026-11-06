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

// ignore_index is allowed to be negative or >= V. Classification rejects every label < 0.
__global__ void sequence_label_count_kernel(const int64_t* labels, int rows, int classes,
                                            int64_t ignore_index, int* n_valid, int* bad) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int64_t label = labels[row];
  if (label == ignore_index) return;
  if (label < 0 || label >= static_cast<int64_t>(classes)) {
    atomicExch(bad, 1);
    return;
  }
  atomicAdd(n_valid, 1);
}

__global__ void sequence_ce_kernel(const float* logits, const int64_t* labels, float* grad,
                                   float* per_sample, int rows, int classes, int64_t ignore_index,
                                   int n_valid, int* bad) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  const float* row_logits = logits + static_cast<int64_t>(row) * classes;
  float* row_grad = grad + static_cast<int64_t>(row) * classes;
  const int64_t label = labels[row];

  if (label == ignore_index) {
    for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
      row_grad[c] = 0.f;
    }
    if (threadIdx.x == 0) per_sample[row] = 0.f;
    return;
  }
  if (label < 0 || label >= static_cast<int64_t>(classes)) {
    for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
      row_grad[c] = 0.f;
    }
    if (threadIdx.x == 0) {
      atomicExch(bad, 1);
      per_sample[row] = 0.f;
    }
    return;
  }

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
  const float inv_sum = 1.f / sum;
  const float inv_n = 1.f / static_cast<float>(n_valid);
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

LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index) {
  int classes = 0;
  int rows = 0;
  if (logits.rank() == 2) {
    rows = logits.shape()[0];
    classes = logits.shape()[1];
    if (labels.shape().size() != 1 || labels.shape()[0] != rows) throw Error("label shape");
  } else if (logits.rank() == 3) {
    if (labels.shape().size() != 2 || labels.shape()[0] != logits.shape()[0] ||
        labels.shape()[1] != logits.shape()[1]) {
      throw Error("label shape");
    }
    classes = logits.shape()[2];
    rows = static_cast<int>(static_cast<int64_t>(logits.shape()[0]) * logits.shape()[1]);
  } else {
    throw Error("logits rank");
  }
  if (classes <= 1) throw Error("logits shape");

  // Scale inside the kernel by 1/n_valid, matching softmax_ce_kernel's 1/N.
  // An empty kept set must return before that division.
  LossResult result{0.f, Tensor(ctx, logits.shape())};
  int* counts = nullptr;
  float* per_sample = nullptr;
  check_cuda(cudaMalloc(&counts, 2 * sizeof(int)), "cudaMalloc");
  int* bad = counts;
  int* n_valid_dev = counts + 1;
  cudaStream_t stream = stream_of(ctx);
  const cudaError_t set = cudaMemsetAsync(counts, 0, 2 * sizeof(int), stream);
  if (set != cudaSuccess) {
    cudaFree(counts);
    check_cuda(set, "cudaMemsetAsync");
  }
  constexpr int kCountThreads = 256;
  const int count_blocks =
      static_cast<int>((static_cast<int64_t>(rows) + kCountThreads - 1) / kCountThreads);
  sequence_label_count_kernel<<<count_blocks, kCountThreads, 0, stream>>>(
      labels.data(), rows, classes, ignore_index, n_valid_dev, bad);
  const cudaError_t count_launch = cudaGetLastError();
  if (count_launch != cudaSuccess) {
    cudaFree(counts);
    check_cuda(count_launch, "sequence_cross_entropy");
  }
  // synchronize() throws through check_cuda. per_sample is still null here.
  const cudaError_t count_sync = cudaStreamSynchronize(stream);
  if (count_sync != cudaSuccess) {
    cudaFree(counts);
    check_cuda(count_sync, "cudaStreamSynchronize");
  }
  int bad_host = 0;
  int n_valid = 0;
  const cudaError_t copy_bad = cudaMemcpy(&bad_host, bad, sizeof(int), cudaMemcpyDeviceToHost);
  if (copy_bad != cudaSuccess) {
    cudaFree(counts);
    check_cuda(copy_bad, "cudaMemcpy");
  }
  const cudaError_t copy_n = cudaMemcpy(&n_valid, n_valid_dev, sizeof(int), cudaMemcpyDeviceToHost);
  if (copy_n != cudaSuccess) {
    cudaFree(counts);
    check_cuda(copy_n, "cudaMemcpy");
  }
  if (bad_host) {
    cudaFree(counts);
    throw Error("label out of range");
  }
  if (n_valid == 0) {
    cudaFree(counts);
    result.grad_logits.zero();
    ctx.synchronize();
    return result;
  }

  const cudaError_t alloc = cudaMalloc(&per_sample, static_cast<size_t>(rows) * sizeof(float));
  if (alloc != cudaSuccess) {
    cudaFree(counts);
    check_cuda(alloc, "cudaMalloc");
  }
  sequence_ce_kernel<<<rows, 128, 0, stream>>>(logits.data(), labels.data(), result.grad_logits.data(),
                                               per_sample, rows, classes, ignore_index, n_valid, bad);
  const cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    cudaFree(per_sample);
    cudaFree(counts);
    check_cuda(launch, "sequence_cross_entropy");
  }

  float sum = 0.f;
  try {
    thrust::device_ptr<float> begin(per_sample);
    sum = thrust::reduce(thrust::cuda::par.on(stream), begin, begin + rows, 0.f);
  } catch (...) {
    cudaFree(per_sample);
    cudaFree(counts);
    throw;
  }
  const cudaError_t loss_sync = cudaStreamSynchronize(stream);
  if (loss_sync != cudaSuccess) {
    cudaFree(per_sample);
    cudaFree(counts);
    check_cuda(loss_sync, "cudaStreamSynchronize");
  }
  const cudaError_t copy = cudaMemcpy(&bad_host, bad, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(per_sample);
  cudaFree(counts);
  check_cuda(copy, "cudaMemcpy");
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n_valid);
  return result;
}

}  // namespace nn
