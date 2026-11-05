#include "nn/tensor.hpp"

#include "check.hpp"
#include "context_impl.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <utility>

namespace nn {
namespace {

int64_t checked_numel(const std::vector<int>& shape) {
  if (shape.empty() || shape.size() > 4) throw Error("tensor rank out of range");
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) throw Error("tensor dimension out of range");
    if (n > static_cast<int64_t>(INT32_MAX) / dim) throw Error("tensor numel out of range");
    n *= dim;
  }
  return n;
}

}  // namespace

Tensor::Tensor(Context& ctx, std::vector<int> shape) : ctx_(&ctx), shape_(std::move(shape)) {
  numel_ = checked_numel(shape_);
  bytes_ = static_cast<size_t>(numel_) * sizeof(float);
  void* raw = nullptr;
  check_cuda(cudaMalloc(&raw, bytes_), "cudaMalloc");
  data_ = static_cast<float*>(raw);
}

void Tensor::release() noexcept {
  if (!data_) return;
  if (ctx_) cudaStreamSynchronize(stream_of(*ctx_));
  cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
  numel_ = 0;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : ctx_(other.ctx_),
      shape_(std::move(other.shape_)),
      numel_(other.numel_),
      data_(other.data_),
      bytes_(other.bytes_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  other.bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  bytes_ = other.bytes_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

void Tensor::fill(float value) {
  thrust::device_ptr<float> ptr(data_);
  thrust::fill(thrust::cuda::par.on(stream_of(*ctx_)), ptr, ptr + numel_, value);
}

void Tensor::zero() { fill(0.f); }

void Tensor::copy_from(const Tensor& src) {
  if (numel_ != src.numel_) throw Error("tensor copy numel mismatch");
  if (&src.context() != ctx_) src.context().synchronize();
  check_cuda(cudaMemcpyAsync(data_, src.data_, bytes_, cudaMemcpyDeviceToDevice, stream_of(*ctx_)),
             "cudaMemcpyAsync");
}

void Tensor::copy_from_host(const float* src, int64_t n) {
  if (n != numel_) throw Error("tensor host copy numel mismatch");
  check_cuda(cudaMemcpyAsync(data_, src, bytes_, cudaMemcpyHostToDevice, stream_of(*ctx_)),
             "cudaMemcpyAsync");
}

void Tensor::copy_from_host(const std::vector<float>& src) {
  copy_from_host(src.data(), static_cast<int64_t>(src.size()));
}

std::vector<float> Tensor::to_host() const {
  ctx_->synchronize();
  std::vector<float> host(static_cast<size_t>(numel_));
  check_cuda(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost), "cudaMemcpy");
  return host;
}

Tensor Tensor::reshape(const std::vector<int>& shape) const {
  if (checked_numel(shape) != numel_) throw Error("tensor reshape numel mismatch");
  Tensor out(*ctx_, shape);
  out.copy_from(*this);
  return out;
}

}  // namespace nn
