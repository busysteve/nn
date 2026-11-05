#include "nn/layers/linear.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <cublas_v2.h>

namespace nn {

Linear::Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng)
    : ctx_(&ctx),
      in_features_(in_features),
      out_features_(out_features),
      weight_(ctx, {out_features, in_features}),
      grad_weight_(ctx, {out_features, in_features}),
      bias_(ctx, {out_features}),
      grad_bias_(ctx, {out_features}) {
  if (in_features < 1 || out_features < 1) throw Error("Linear size");
  kaiming_normal(weight_, in_features, rng);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
}

std::vector<NamedParam> Linear::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

void Linear::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

const Tensor& Linear::forward(const Tensor& input) {
  if (input.rank() != 2 || input.shape()[1] != in_features_) throw Error("Linear input shape");
  const int n = input.shape()[0];
  ensure_tensor(output_, *ctx_, {n, out_features_});
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);

  // Row-major y = x W^T. cuBLAS is column-major, so this is C = W x^T
  // with the leading dimensions equal to the row-major inner sizes.
  const float one = 1.f;
  const float zero = 0.f;
  check_cublas(cublasGemmEx(cublas_of(*ctx_), CUBLAS_OP_T, CUBLAS_OP_N, out_features_, n,
                            in_features_, &one, weight_.data(), CUDA_R_32F, in_features_,
                            input.data(), CUDA_R_32F, in_features_, &zero, output_->data(),
                            CUDA_R_32F, out_features_, CUBLAS_COMPUTE_32F_PEDANTIC,
                            CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  launch_add_bias(stream_of(*ctx_), output_->data(), bias_.data(), n, out_features_);
  return *output_;
}

const Tensor& Linear::backward(const Tensor& grad_output) {
  if (!cache_x_ || grad_output.rank() != 2 || grad_output.shape()[0] != cache_x_->shape()[0] ||
      grad_output.shape()[1] != out_features_) {
    throw Error("Linear backward shape");
  }
  const int n = grad_output.shape()[0];
  ensure_tensor(grad_input_, *ctx_, {n, in_features_});
  const float one = 1.f;
  const float zero = 0.f;
  cublasHandle_t handle = cublas_of(*ctx_);
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features_, n, out_features_, &one,
                            weight_.data(), CUDA_R_32F, in_features_, grad_output.data(),
                            CUDA_R_32F, out_features_, &zero, grad_input_->data(), CUDA_R_32F,
                            in_features_, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features_, out_features_, n, &one,
                            cache_x_->data(), CUDA_R_32F, in_features_, grad_output.data(),
                            CUDA_R_32F, out_features_, &one, grad_weight_.data(), CUDA_R_32F,
                            in_features_, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
  launch_bias_grad(stream_of(*ctx_), grad_output.data(), grad_bias_.data(), n, out_features_);
  return *grad_input_;
}

}  // namespace nn
