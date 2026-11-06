#include "nn/layers/relu.hpp"

#include "context_impl.hpp"
#include "descriptors.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct ReLU::Impl {
  TensorDesc x_desc;
  TensorDesc y_desc;
  ActDesc act;
};

ReLU::ReLU(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

ReLU::~ReLU() = default;

std::vector<NamedParam> ReLU::parameters() const { return {}; }

void ReLU::zero_grad() {}

namespace {

void set_rank(TensorDesc& desc, const std::vector<int>& shape) {
  if (shape.size() == 2) {
    desc.set4(shape[0], 1, 1, shape[1]);
  } else if (shape.size() == 3) {
    // (N, T, F) matches NCHW (N, 1, T, F) element order. No copy.
    desc.set4(shape[0], 1, shape[1], shape[2]);
  } else if (shape.size() == 4) {
    desc.set4(shape[0], shape[1], shape[2], shape[3]);
  } else {
    throw Error("ReLU rank");
  }
}

}  // namespace

const Tensor& ReLU::forward(const Tensor& input) {
  set_rank(impl_->x_desc, input.shape());
  set_rank(impl_->y_desc, input.shape());
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const float one = 1.f;
  const float zero = 0.f;
  check_cudnn(cudnnActivationForward(cudnn_of(*ctx_), impl_->act.raw, &one, impl_->x_desc.raw,
                                     input.data(), &zero, impl_->y_desc.raw, output_->data()),
              "cudnnActivationForward");
  return *output_;
}

const Tensor& ReLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("ReLU backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  const float one = 1.f;
  const float zero = 0.f;
  check_cudnn(cudnnActivationBackward(cudnn_of(*ctx_), impl_->act.raw, &one, impl_->y_desc.raw,
                                      output_->data(), impl_->y_desc.raw, grad_output.data(),
                                      impl_->x_desc.raw, cache_x_->data(), &zero, impl_->x_desc.raw,
                                      grad_input_->data()),
              "cudnnActivationBackward");
  return *grad_input_;
}

}  // namespace nn
