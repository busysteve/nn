#include "nn/layers/multi_head_attention.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"

#include <cublas_v2.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn {
namespace {

constexpr int kThreads = 256;
constexpr float kMask = -1.0e9f;

// Trailing sync: the next reduction reuses the same shared buffer.
__device__ float block_sum(float value, float* buffer) {
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

__device__ float block_max(float value, float* buffer) {
  const int tid = static_cast<int>(threadIdx.x);
  buffer[tid] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] = fmaxf(buffer[tid], buffer[tid + offset]);
    __syncthreads();
  }
  const float total = buffer[0];
  __syncthreads();
  return total;
}

// (N,T,H,Dh) and (N,H,T,Dh) are not the same strides. reshape() would only copy.
__global__ void split_heads_kernel(float* dst, const float* src, int time, int heads, int dh,
                                   int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t d = i % dh;
  const int64_t h = (i / dh) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
  const int64_t o = ((batch * heads + h) * time + t) * dh + d;
  dst[o] = src[i];
}

__global__ void merge_heads_kernel(float* dst, const float* src, int time, int heads, int dh,
                                   int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t d = i % dh;
  const int64_t h = (i / dh) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
  const int64_t s = ((batch * heads + h) * time + t) * dh + d;
  dst[i] = src[s];
}

__global__ void causal_mask_kernel(float* scores, int time, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int col = static_cast<int>(i % time);
  const int row = static_cast<int>((i / time) % time);
  if (col > row) scores[i] = kMask;
}

__global__ void softmax_rows_kernel(float* out, const float* in, int cols) {
  __shared__ float buffer[kThreads];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float* row_in = in + base;
  float* row_out = out + base;

  float max_v = -INFINITY;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    max_v = fmaxf(max_v, row_in[col]);
  }
  max_v = block_max(max_v, buffer);

  float sum = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    sum += expf(row_in[col] - max_v);
  }
  sum = block_sum(sum, buffer);
  const float inv = 1.f / sum;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    row_out[col] = expf(row_in[col] - max_v) * inv;
  }
}

__global__ void softmax_backward_kernel(float* ds, const float* probs, const float* dp, int cols,
                                        int causal) {
  __shared__ float buffer[kThreads];
  const int row = static_cast<int>(blockIdx.x);
  const int query = row % cols;
  const int64_t base = static_cast<int64_t>(row) * cols;

  float dot = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    dot += dp[base + col] * probs[base + col];
  }
  dot = block_sum(dot, buffer);

  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    float grad = probs[base + col] * (dp[base + col] - dot);
    // The mask overwrote the score with a constant, so that score has no gradient.
    if (causal && col > query) grad = 0.f;
    ds[base + col] = grad;
  }
}

int blocks_for(int64_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

void launch_split_heads(cudaStream_t stream, float* dst, const float* src, int time, int heads,
                        int dh, int64_t n) {
  split_heads_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, heads, dh, n);
  check_cuda(cudaGetLastError(), "mha_split_heads");
}

void launch_merge_heads(cudaStream_t stream, float* dst, const float* src, int time, int heads,
                        int dh, int64_t n) {
  merge_heads_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, heads, dh, n);
  check_cuda(cudaGetLastError(), "mha_merge_heads");
}

void launch_causal_mask(cudaStream_t stream, float* scores, int time, int64_t n) {
  causal_mask_kernel<<<blocks_for(n), kThreads, 0, stream>>>(scores, time, n);
  check_cuda(cudaGetLastError(), "mha_causal_mask");
}

void launch_softmax_rows(cudaStream_t stream, float* out, const float* in, int rows, int cols) {
  softmax_rows_kernel<<<rows, kThreads, 0, stream>>>(out, in, cols);
  check_cuda(cudaGetLastError(), "mha_softmax");
}

void launch_softmax_backward(cudaStream_t stream, float* ds, const float* probs, const float* dp,
                             int rows, int cols, bool causal) {
  softmax_backward_kernel<<<rows, kThreads, 0, stream>>>(ds, probs, dp, cols, causal ? 1 : 0);
  check_cuda(cudaGetLastError(), "mha_softmax_backward");
}

// Row-major (T, Dh) occupies the same buffer as column-major (Dh, T).
// S = scale * Q K^T  <=>  S_cm = scale * K_cm^T Q_cm.
void gemm_scores(cublasHandle_t handle, const float* q, const float* k, float* scores, int time,
                 int dh, int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_cublas(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, time, time, dh, &scale,
                                          k, CUDA_R_32F, dh, stride_qk, q, CUDA_R_32F, dh,
                                          stride_qk, &beta, scores, CUDA_R_32F, time, stride_s,
                                          batch, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmStridedBatchedEx");
}

// dQ = scale * dS K.  dQ_cm = scale * K_cm dS_cm.
void gemm_dq(cublasHandle_t handle, const float* k, const float* ds, float* dq, int time, int dh,
             int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_cublas(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dh, time, time, &scale,
                                          k, CUDA_R_32F, dh, stride_qk, ds, CUDA_R_32F, time,
                                          stride_s, &beta, dq, CUDA_R_32F, dh, stride_qk, batch,
                                          CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmStridedBatchedEx");
}

// dK = scale * dS^T Q.  dK_cm = scale * Q_cm dS_cm^T.
void gemm_dk(cublasHandle_t handle, const float* q, const float* ds, float* dk, int time, int dh,
             int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_cublas(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, dh, time, time, &scale,
                                          q, CUDA_R_32F, dh, stride_qk, ds, CUDA_R_32F, time,
                                          stride_s, &beta, dk, CUDA_R_32F, dh, stride_qk, batch,
                                          CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmStridedBatchedEx");
}

void append_params(std::vector<NamedParam>& out, const Layer& layer, const char* prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = std::string(prefix) + param.name;
    out.push_back(param);
  }
}

Context* attention_context(Context& ctx, int d_model, int num_heads, bool use_rope) {
  // RoPE is a later change. Reject it here so the flag cannot be stored and ignored.
  if (use_rope) throw std::invalid_argument("use_rope");
  if (d_model < 1 || num_heads < 1 || d_model % num_heads != 0) {
    throw Error("MultiHeadAttention size");
  }
  return &ctx;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal,
                                       bool use_bias, bool use_rope, std::mt19937& rng)
    : ctx_(attention_context(ctx, d_model, num_heads, use_rope)),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(d_model / num_heads),
      causal_(causal),
      q_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      k_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      v_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)),
      out_(std::make_unique<Linear>(*ctx_, d_model, d_model, use_bias, rng)) {}

std::vector<NamedParam> MultiHeadAttention::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *q_, "q.");
  append_params(params, *k_, "k.");
  append_params(params, *v_, "v.");
  append_params(params, *out_, "out.");
  return params;
}

void MultiHeadAttention::zero_grad() {
  q_->zero_grad();
  k_->zero_grad();
  v_->zero_grad();
  out_->zero_grad();
}

const Tensor& MultiHeadAttention::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("MultiHeadAttention input shape");
  const int n = shape[0];
  const int time = shape[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};

  ensure_tensor(q_heads_, *ctx_, heads_shape);
  ensure_tensor(k_heads_, *ctx_, heads_shape);
  ensure_tensor(v_heads_, *ctx_, heads_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, heads_shape);
  ensure_tensor(merged_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);

  cudaStream_t stream = stream_of(*ctx_);
  cublasHandle_t handle = cublas_of(*ctx_);
  const int64_t hidden = input.numel();
  launch_split_heads(stream, q_heads_->data(), q_->forward(input).data(), time, heads, dh, hidden);
  launch_split_heads(stream, k_heads_->data(), k_->forward(input).data(), time, heads, dh, hidden);
  launch_split_heads(stream, v_heads_->data(), v_->forward(input).data(), time, heads, dh, hidden);

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  gemm_scores(handle, q_heads_->data(), k_heads_->data(), scores_->data(), time, dh, batch, scale);
  // -1e9, not -inf: a row of infinities makes max-subtract softmax NaN.
  if (causal_) launch_causal_mask(stream, scores_->data(), time, scores_->numel());
  launch_softmax_rows(stream, attn_->data(), scores_->data(), batch * time, time);
  // context = attn V, same row-major product as dQ = dS K with scale 1.
  gemm_dq(handle, v_heads_->data(), attn_->data(), context_->data(), time, dh, batch, 1.f);
  launch_merge_heads(stream, merged_->data(), context_->data(), time, heads, dh, hidden);
  output_->copy_from(out_->forward(*merged_));
  return *output_;
}

const Tensor& MultiHeadAttention::backward(const Tensor& grad_output) {
  if (!output_ || !attn_ || !q_heads_ || !k_heads_ || !v_heads_ ||
      grad_output.shape() != output_->shape()) {
    throw Error("MultiHeadAttention backward shape");
  }
  const int n = grad_output.shape()[0];
  const int time = grad_output.shape()[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int>& flat_shape = grad_output.shape();

  ensure_tensor(d_context_, *ctx_, heads_shape);
  ensure_tensor(d_attn_, *ctx_, score_shape);
  ensure_tensor(d_scores_, *ctx_, score_shape);
  ensure_tensor(d_q_, *ctx_, heads_shape);
  ensure_tensor(d_k_, *ctx_, heads_shape);
  ensure_tensor(d_v_, *ctx_, heads_shape);
  ensure_tensor(merged_grad_, *ctx_, flat_shape);
  ensure_tensor(grad_pair_, *ctx_, flat_shape);
  ensure_tensor(grad_input_, *ctx_, flat_shape);

  cudaStream_t stream = stream_of(*ctx_);
  cublasHandle_t handle = cublas_of(*ctx_);
  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));

  const Tensor& d_merged = out_->backward(grad_output);
  launch_split_heads(stream, d_context_->data(), d_merged.data(), time, heads, dh, hidden);
  // dAttn = dContext V^T, the same product as scores = Q K^T.
  gemm_scores(handle, d_context_->data(), v_heads_->data(), d_attn_->data(), time, dh, batch, 1.f);
  // dV = attn^T dContext, the same product as dK = dS^T Q.
  gemm_dk(handle, d_context_->data(), attn_->data(), d_v_->data(), time, dh, batch, 1.f);
  launch_softmax_backward(stream, d_scores_->data(), attn_->data(), d_attn_->data(), batch * time,
                          time, causal_);
  gemm_dq(handle, k_heads_->data(), d_scores_->data(), d_q_->data(), time, dh, batch, scale);
  gemm_dk(handle, q_heads_->data(), d_scores_->data(), d_k_->data(), time, dh, batch, scale);

  launch_merge_heads(stream, merged_grad_->data(), d_q_->data(), time, heads, dh, hidden);
  const Tensor& gq = q_->backward(*merged_grad_);
  launch_merge_heads(stream, merged_grad_->data(), d_k_->data(), time, heads, dh, hidden);
  const Tensor& gk = k_->backward(*merged_grad_);
  launch_merge_heads(stream, merged_grad_->data(), d_v_->data(), time, heads, dh, hidden);
  const Tensor& gv = v_->backward(*merged_grad_);

  launch_add2(stream, grad_pair_->data(), gq.data(), gk.data(), hidden);
  launch_add2(stream, grad_input_->data(), grad_pair_->data(), gv.data(), hidden);
  return *grad_input_;
}

}  // namespace nn
