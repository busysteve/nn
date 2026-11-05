#include "nn/layers/conv2d.hpp"

#include "context_impl.hpp"
#include "descriptors.hpp"
#include "ensure.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <algorithm>
#include <string>

namespace nn {
namespace {

constexpr cudnnConvolutionFwdAlgo_t kFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
constexpr cudnnConvolutionBwdFilterAlgo_t kBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
constexpr cudnnConvolutionBwdDataAlgo_t kBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
constexpr size_t kMaxWorkspace = 64ull * 1024ull * 1024ull;

}  // namespace

struct Conv2d::Impl {
  TensorDesc x_desc;
  TensorDesc y_desc;
  TensorDesc bias_desc;
  FilterDesc filter;
  ConvDesc conv;
};

Conv2d::Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
               bool use_bias, std::mt19937& rng)
    : ctx_(&ctx),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_(kernel),
      stride_(stride),
      pad_(pad),
      use_bias_(use_bias),
      weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      grad_weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      impl_(std::make_unique<Impl>()) {
  if (in_channels < 1 || out_channels < 1) throw Error("Conv2d channels");
  conv_out_size(1 + kernel, kernel, stride, pad);
  impl_->filter.set(out_channels, in_channels, kernel);
  impl_->conv.set(pad, stride);
  kaiming_normal(weight_, in_channels * kernel * kernel, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_channels});
    grad_bias_.emplace(ctx, std::vector<int>{out_channels});
    bias_->zero();
    grad_bias_->zero();
    impl_->bias_desc.set4(1, out_channels, 1, 1);
  }
}

Conv2d::~Conv2d() {
  if (workspace_) {
    cudaStreamSynchronize(stream_of(*ctx_));
    cudaFree(workspace_);
  }
}

std::vector<NamedParam> Conv2d::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Conv2d::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

void Conv2d::prepare(int n, int h, int w) {
  const int ho = conv_out_size(h, kernel_, stride_, pad_);
  const int wo = conv_out_size(w, kernel_, stride_, pad_);
  impl_->x_desc.set4(n, in_channels_, h, w);
  impl_->y_desc.set4(n, out_channels_, ho, wo);
  int cn = 0, cc = 0, ch = 0, cw = 0;
  check_cudnn(cudnnGetConvolution2dForwardOutputDim(impl_->conv.raw, impl_->x_desc.raw,
                                                    impl_->filter.raw, &cn, &cc, &ch, &cw),
              "cudnnGetConvolution2dForwardOutputDim");
  if (cn != n || cc != out_channels_ || ch != ho || cw != wo) throw Error("Conv2d output dim mismatch");

  cudnnHandle_t handle = cudnn_of(*ctx_);
  size_t forward_bytes = 0;
  size_t data_bytes = 0;
  size_t filter_bytes = 0;
  check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, impl_->x_desc.raw, impl_->filter.raw,
                                                      impl_->conv.raw, impl_->y_desc.raw, kFwdAlgo,
                                                      &forward_bytes),
              "cudnnGetConvolutionForwardWorkspaceSize");
  check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, impl_->filter.raw, impl_->y_desc.raw,
                                                           impl_->conv.raw, impl_->x_desc.raw,
                                                           kBwdDataAlgo, &data_bytes),
              "cudnnGetConvolutionBackwardDataWorkspaceSize");
  check_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                  handle, impl_->x_desc.raw, impl_->y_desc.raw, impl_->conv.raw, impl_->filter.raw,
                  kBwdFilterAlgo, &filter_bytes),
              "cudnnGetConvolutionBackwardFilterWorkspaceSize");
  const size_t need = std::max(forward_bytes, std::max(data_bytes, filter_bytes));
  if (need > kMaxWorkspace) {
    throw Error("conv workspace " + std::to_string(need) + " exceeds 64 MiB budget");
  }
  if (need > workspace_bytes_) {
    if (workspace_) {
      cudaStreamSynchronize(stream_of(*ctx_));
      cudaFree(workspace_);
      workspace_ = nullptr;
    }
    if (need > 0) {
      check_cuda(cudaMalloc(&workspace_, need), "cudaMalloc");
      workspace_bytes_ = need;
    }
  }
  ensure_tensor(output_, *ctx_, {n, out_channels_, ho, wo});
  ensure_tensor(grad_input_, *ctx_, {n, in_channels_, h, w});
}

const Tensor& Conv2d::forward(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != in_channels_) throw Error("Conv2d input shape");
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  prepare(n, h, w);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const float one = 1.f;
  const float zero = 0.f;
  check_cudnn(cudnnConvolutionForward(cudnn_of(*ctx_), &one, impl_->x_desc.raw, input.data(),
                                      impl_->filter.raw, weight_.data(), impl_->conv.raw, kFwdAlgo,
                                      workspace_, workspace_bytes_, &zero, impl_->y_desc.raw,
                                      output_->data()),
              "cudnnConvolutionForward");
  if (bias_) {
    check_cudnn(cudnnAddTensor(cudnn_of(*ctx_), &one, impl_->bias_desc.raw, bias_->data(), &one,
                               impl_->y_desc.raw, output_->data()),
                "cudnnAddTensor");
  }
  return *output_;
}

const Tensor& Conv2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("Conv2d backward shape");
  }
  const float one = 1.f;
  const float zero = 0.f;
  cudnnHandle_t handle = cudnn_of(*ctx_);
  check_cudnn(cudnnConvolutionBackwardData(handle, &one, impl_->filter.raw, weight_.data(),
                                           impl_->y_desc.raw, grad_output.data(), impl_->conv.raw,
                                           kBwdDataAlgo, workspace_, workspace_bytes_, &zero,
                                           impl_->x_desc.raw, grad_input_->data()),
              "cudnnConvolutionBackwardData");
  check_cudnn(cudnnConvolutionBackwardFilter(handle, &one, impl_->x_desc.raw, cache_x_->data(),
                                             impl_->y_desc.raw, grad_output.data(), impl_->conv.raw,
                                             kBwdFilterAlgo, workspace_, workspace_bytes_, &one,
                                             impl_->filter.raw, grad_weight_.data()),
              "cudnnConvolutionBackwardFilter");
  if (grad_bias_) {
    check_cudnn(cudnnConvolutionBackwardBias(handle, &one, impl_->y_desc.raw, grad_output.data(),
                                             &one, impl_->bias_desc.raw, grad_bias_->data()),
                "cudnnConvolutionBackwardBias");
  }
  return *grad_input_;
}

}  // namespace nn
