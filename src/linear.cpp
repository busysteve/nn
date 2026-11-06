#include "nn/layers/linear.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <cublas_v2.h>

#include <cstdint>

namespace nn {
namespace {

int gemm_rows(const std::vector<int>& shape) {
  if (shape.size() == 2) return shape[0];
  // Rank 3 stays (N, T, F). rows is N*T on data(); reshape() would copy.
  if (shape.size() != 3) throw Error("Linear input shape");
  return static_cast<int>(static_cast<int64_t>(shape[0]) * shape[1]);
}

}  // namespace

Linear::Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng)
    : Linear(ctx, in_features, out_features, true, rng) {}

Linear::Linear(Context& ctx, int in_features, int out_features, bool use_bias, std::mt19937& rng)
    : ctx_(&ctx),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(ctx, {out_features, in_features}),
      grad_weight_(ctx, {out_features, in_features}) {
  if (in_features < 1 || out_features < 1) throw Error("Linear size");
  kaiming_normal(weight_, in_features, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_features});
    grad_bias_.emplace(ctx, std::vector<int>{out_features});
    bias_->zero();
    grad_bias_->zero();
  }
}

std::vector<NamedParam> Linear::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(
      NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(
        NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Linear::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

const Tensor& Linear::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != in_features_) {
    throw Error("Linear input shape");
  }
  const int rows = gemm_rows(shape);
  std::vector<int> out_shape = shape;
  out_shape.back() = out_features_;
  ensure_tensor(output_, *ctx_, out_shape);
  ensure_tensor(cache_x_, *ctx_, shape);
  cache_x_->copy_from(input);

  // Row-major y = x W^T. cuBLAS is column-major, so this is C = W x^T
  // with the leading dimensions equal to the row-major inner sizes.
  const float one = 1.f;
  const float zero = 0.f;
  check_cublas(cublasGemmEx(cublas_of(*ctx_), CUBLAS_OP_T, CUBLAS_OP_N, out_features_, rows,
                            in_features_, &one, weight_.data(), CUDA_R_32F, in_features_,
                            input.data(), CUDA_R_32F, in_features_, &zero, output_->data(),
                            CUDA_R_32F, out_features_, CUBLAS_COMPUTE_32F_PEDANTIC,
                            CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  if (bias_) launch_add_bias(stream_of(*ctx_), output_->data(), bias_->data(), rows, out_features_);
  return *output_;
}

const Tensor& Linear::backward(const Tensor& grad_output) {
  if (!cache_x_ || grad_output.shape().size() != cache_x_->shape().size() ||
      grad_output.shape().back() != out_features_) {
    throw Error("Linear backward shape");
  }
  const std::vector<int>& in_shape = cache_x_->shape();
  for (size_t axis = 0; axis + 1 < in_shape.size(); ++axis) {
    if (grad_output.shape()[axis] != in_shape[axis]) throw Error("Linear backward shape");
  }
  if (in_shape.size() != 2 && in_shape.size() != 3) throw Error("Linear backward shape");
  const int rows = gemm_rows(in_shape);
  ensure_tensor(grad_input_, *ctx_, in_shape);
  const float one = 1.f;
  const float zero = 0.f;
  cublasHandle_t handle = cublas_of(*ctx_);
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features_, rows, out_features_,
                            &one, weight_.data(), CUDA_R_32F, in_features_, grad_output.data(),
                            CUDA_R_32F, out_features_, &zero, grad_input_->data(), CUDA_R_32F,
                            in_features_, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features_, out_features_, rows,
                            &one, cache_x_->data(), CUDA_R_32F, in_features_, grad_output.data(),
                            CUDA_R_32F, out_features_, &one, grad_weight_.data(), CUDA_R_32F,
                            in_features_, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  if (grad_bias_) {
    launch_bias_grad(stream_of(*ctx_), grad_output.data(), grad_bias_->data(), rows, out_features_);
  }
  return *grad_input_;
}

}  // namespace nn
