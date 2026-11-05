#include "nn/optim.hpp"

#include "context_impl.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"

namespace nn {

Sgd::Sgd(std::vector<NamedParam> params, float learning_rate, float momentum)
    : params_(std::move(params)), lr_(learning_rate), momentum_(momentum) {
  if (params_.empty()) throw Error("SGD has no parameters");
  Context& ctx = params_.front().value->context();
  velocity_.reserve(params_.size());
  for (const NamedParam& param : params_) {
    velocity_.emplace_back(ctx, param.value->shape());
    velocity_.back().zero();
  }
}

void Sgd::step() {
  cudaStream_t stream = stream_of(params_.front().value->context());
  for (size_t i = 0; i < params_.size(); ++i) {
    launch_sgd(stream, params_[i].value->data(), velocity_[i].data(), params_[i].grad->data(),
               params_[i].value->numel(), lr_, momentum_);
  }
  ++step_;
}

}  // namespace nn
