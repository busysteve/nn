#include "nn/layers/global_avg_pool2d.hpp"

#include "context_impl.hpp"
#include "descriptors.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct GlobalAvgPool2d::Impl {
  TensorDesc x_desc;
  TensorDesc y_desc;
  PoolDesc pool;
};

GlobalAvgPool2d::GlobalAvgPool2d(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

GlobalAvgPool2d::~GlobalAvgPool2d() = default;

std::vector<NamedParam> GlobalAvgPool2d::parameters() const { return {}; }

void GlobalAvgPool2d::zero_grad() {}

const Tensor& GlobalAvgPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("GlobalAvgPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  impl_->pool.set_global_avg(h, w);
  impl_->x_desc.set4(n, c, h, w);
  impl_->y_desc.set4(n, c, 1, 1);
  int on = 0, oc = 0, oh = 0, ow = 0;
  check_cudnn(cudnnGetPooling2dForwardOutputDim(impl_->pool.raw, impl_->x_desc.raw, &on, &oc, &oh, &ow),
              "cudnnGetPooling2dForwardOutputDim");
  if (on != n || oc != c || oh != 1 || ow != 1) throw Error("GlobalAvgPool2d output dim mismatch");
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, c, 1, 1});
  ensure_tensor(cache_y_, *ctx_, output_->shape());
  cache_x_->copy_from(input);
  const float one = 1.f;
  const float zero = 0.f;
  check_cudnn(cudnnPoolingForward(cudnn_of(*ctx_), impl_->pool.raw, &one, impl_->x_desc.raw,
                                  input.data(), &zero, impl_->y_desc.raw, output_->data()),
              "cudnnPoolingForward");
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& GlobalAvgPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("GlobalAvgPool2d backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  const float one = 1.f;
  const float zero = 0.f;
  check_cudnn(cudnnPoolingBackward(cudnn_of(*ctx_), impl_->pool.raw, &one, impl_->y_desc.raw,
                                   cache_y_->data(), impl_->y_desc.raw, grad_output.data(),
                                   impl_->x_desc.raw, cache_x_->data(), &zero, impl_->x_desc.raw,
                                   grad_input_->data()),
              "cudnnPoolingBackward");
  return *grad_input_;
}

}  // namespace nn
