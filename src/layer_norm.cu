#include "nn/layers/layer_norm.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

#include <cmath>
#include <vector>

namespace nn {
namespace {

constexpr int kThreads = 256;

// Trailing sync: every thread must finish reading the total before the next call reuses the buffer.
__forceinline__ __device__ float block_sum(float value, float* buffer) {
  const int tid = static_cast<int>(threadIdx.x);
  buffer[tid] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    __syncthreads();
  }
  const float total = buffer[0];
  __syncthreads();
  return total;
}

__global__ void layer_norm_forward_kernel(float* y, float* mean_out, float* inv_std_out,
                                          const float* x, const float* gamma, const float* beta,
                                          int cols, float eps) {
  __shared__ float buffer[kThreads];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float* row_x = x + base;
  float* row_y = y + base;

  float sum = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) sum += row_x[col];
  const float mean = block_sum(sum, buffer) / static_cast<float>(cols);

  float var_sum = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    const float diff = row_x[col] - mean;
    var_sum += diff * diff;
  }
  // Population second moment: divide by D, not D - 1.
  var_sum = block_sum(var_sum, buffer);
  const float inv_std = rsqrtf(var_sum / static_cast<float>(cols) + eps);

  if (threadIdx.x == 0) {
    mean_out[row] = mean;
    inv_std_out[row] = inv_std;
  }
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    const float xhat = (row_x[col] - mean) * inv_std;
    row_y[col] = xhat * gamma[col] + beta[col];
  }
}

__global__ void layer_norm_backward_input_kernel(float* dx, const float* dy, const float* x,
                                                 const float* mean, const float* inv_std,
                                                 const float* gamma, int cols) {
  __shared__ float buffer[kThreads];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float mean_v = mean[row];
  const float inv = inv_std[row];

  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    sum_dhat += dhat;
    sum_dhat_xhat += dhat * xhat;
  }
  sum_dhat = block_sum(sum_dhat, buffer);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer);
  const float mean_dhat = sum_dhat / static_cast<float>(cols);
  const float mean_dhat_xhat = sum_dhat_xhat / static_cast<float>(cols);

  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    dx[base + col] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
  }
}

__global__ void layer_norm_param_grad_kernel(const float* dy, const float* x, const float* mean,
                                             const float* inv_std, float* grad_gamma,
                                             float* grad_beta, int rows, int cols) {
  __shared__ float buffer[kThreads];
  const int col = static_cast<int>(blockIdx.x);
  float sum_gamma = 0.f;
  float sum_beta = 0.f;
  for (int row = static_cast<int>(threadIdx.x); row < rows; row += kThreads) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float xhat = (x[index] - mean[row]) * inv_std[row];
    const float dy_v = dy[index];
    sum_gamma += dy_v * xhat;
    sum_beta += dy_v;
  }
  sum_gamma = block_sum(sum_gamma, buffer);
  sum_beta = block_sum(sum_beta, buffer);
  if (threadIdx.x == 0) {
    grad_gamma[col] += sum_gamma;
    grad_beta[col] += sum_beta;
  }
}

int normalized_rows(const Tensor& input, int normalized_shape) {
  const std::vector<int>& shape = input.shape();
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != normalized_shape) {
    throw Error("LayerNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_layer_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("LayerNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("LayerNorm eps");
  return ctx;
}

}  // namespace

LayerNorm::LayerNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_layer_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}),
      bias_(ctx, {normalized_shape}),
      grad_bias_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
}

std::vector<NamedParam> LayerNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

void LayerNorm::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

const Tensor& LayerNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_tensor(cache_x_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);
  ensure_tensor(mean_, *ctx_, std::vector<int>{rows});
  ensure_tensor(inv_std_, *ctx_, std::vector<int>{rows});
  cache_x_->copy_from(input);
  layer_norm_forward_kernel<<<rows, kThreads, 0, stream_of(*ctx_)>>>(
      output_->data(), mean_->data(), inv_std_->data(), cache_x_->data(), weight_.data(),
      bias_.data(), normalized_shape_, eps_);
  check_cuda(cudaGetLastError(), "layer_norm_forward");
  return *output_;
}

const Tensor& LayerNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !mean_ || !inv_std_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("LayerNorm backward shape");
  }
  const int rows = static_cast<int>(mean_->numel());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  cudaStream_t stream = stream_of(*ctx_);
  layer_norm_backward_input_kernel<<<rows, kThreads, 0, stream>>>(
      grad_input_->data(), grad_output.data(), cache_x_->data(), mean_->data(), inv_std_->data(),
      weight_.data(), normalized_shape_);
  check_cuda(cudaGetLastError(), "layer_norm_backward_input");
  layer_norm_param_grad_kernel<<<normalized_shape_, kThreads, 0, stream>>>(
      grad_output.data(), cache_x_->data(), mean_->data(), inv_std_->data(), grad_weight_.data(),
      grad_bias_.data(), rows, normalized_shape_);
  check_cuda(cudaGetLastError(), "layer_norm_backward_param");
  return *grad_input_;
}

}  // namespace nn
