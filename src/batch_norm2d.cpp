#include "nn/layers/batch_norm2d.hpp"

#include "context_impl.hpp"
#include "descriptors.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct BatchNorm2d::Impl {
  TensorDesc x_desc;
  TensorDesc bn_desc;
};

BatchNorm2d::BatchNorm2d(Context& ctx, int channels)
    : ctx_(&ctx),
      channels_(channels),
      weight_(ctx, {channels}),
      grad_weight_(ctx, {channels}),
      bias_(ctx, {channels}),
      grad_bias_(ctx, {channels}),
      running_mean_(ctx, {channels}),
      running_var_(ctx, {channels}),
      saved_mean_(ctx, {channels}),
      saved_inv_variance_(ctx, {channels}),
      impl_(std::make_unique<Impl>()) {
  if (channels < 1) throw Error("BatchNorm2d channels");
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
  running_mean_.zero();
  running_var_.fill(1.f);
  saved_mean_.zero();
  saved_inv_variance_.zero();
}

BatchNorm2d::~BatchNorm2d() = default;

std::vector<NamedParam> BatchNorm2d::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

std::vector<NamedBuffer> BatchNorm2d::buffers() const {
  auto* self = const_cast<BatchNorm2d*>(this);
  return {NamedBuffer{"running_mean", &self->running_mean_, &self->batches_},
          NamedBuffer{"running_var", &self->running_var_, &self->batches_}};
}

void BatchNorm2d::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

void BatchNorm2d::set_training(bool training) { training_ = training; }

void BatchNorm2d::bind(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != channels_) throw Error("BatchNorm2d input shape");
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  impl_->x_desc.set4(n, channels_, h, w);
  check_cudnn(cudnnDeriveBNTensorDescriptor(impl_->bn_desc.raw, impl_->x_desc.raw,
                                            CUDNN_BATCHNORM_SPATIAL),
              "cudnnDeriveBNTensorDescriptor");
  ensure_tensor(output_, *ctx_, input.shape());
  ensure_tensor(grad_input_, *ctx_, input.shape());
}

const Tensor& BatchNorm2d::forward(const Tensor& input) {
  bind(input);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const float one = 1.f;
  const float zero = 0.f;
  const double eps = 1e-5;
  if (training_) {
    const int64_t count =
        static_cast<int64_t>(input.shape()[0]) * input.shape()[2] * input.shape()[3];
    if (count <= 1) throw Error("BatchNorm train mode needs more than one value per channel");
    // Cumulative moving average. cuDNN applies running = factor * batch + (1 - factor) * running.
    ++batches_;
    const double factor = 1.0 / static_cast<double>(batches_);
    check_cudnn(cudnnBatchNormalizationForwardTraining(
                    cudnn_of(*ctx_), CUDNN_BATCHNORM_SPATIAL, &one, &zero, impl_->x_desc.raw,
                    input.data(), impl_->x_desc.raw, output_->data(), impl_->bn_desc.raw,
                    weight_.data(), bias_.data(), factor, running_mean_.data(), running_var_.data(),
                    eps, saved_mean_.data(), saved_inv_variance_.data()),
                "cudnnBatchNormalizationForwardTraining");
  } else {
    check_cudnn(cudnnBatchNormalizationForwardInference(
                    cudnn_of(*ctx_), CUDNN_BATCHNORM_SPATIAL, &one, &zero, impl_->x_desc.raw,
                    input.data(), impl_->x_desc.raw, output_->data(), impl_->bn_desc.raw,
                    weight_.data(), bias_.data(), running_mean_.data(), running_var_.data(), eps),
                "cudnnBatchNormalizationForwardInference");
  }
  return *output_;
}

const Tensor& BatchNorm2d::backward(const Tensor& grad_output) {
  if (!training_) throw Error("BatchNorm backward requires train mode");
  if (!cache_x_ || grad_output.shape() != cache_x_->shape()) throw Error("BatchNorm2d backward shape");
  const float one = 1.f;
  const float zero = 0.f;
  const double eps = 1e-5;
  check_cudnn(cudnnBatchNormalizationBackward(
                  cudnn_of(*ctx_), CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &one,
                  impl_->x_desc.raw, cache_x_->data(), impl_->x_desc.raw, grad_output.data(),
                  impl_->x_desc.raw, grad_input_->data(), impl_->bn_desc.raw, weight_.data(),
                  grad_weight_.data(), grad_bias_.data(), eps, saved_mean_.data(),
                  saved_inv_variance_.data()),
              "cudnnBatchNormalizationBackward");
  return *grad_input_;
}

}  // namespace nn
