#include "context_impl.hpp"
#include "check.hpp"

namespace nn {

Context::Context() : impl_(std::make_unique<Impl>()) {
  check_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
  check_cudnn(cudnnCreate(&impl_->cudnn), "cudnnCreate");
  check_cudnn(cudnnSetStream(impl_->cudnn, impl_->stream), "cudnnSetStream");
  check_cublas(cublasCreate(&impl_->cublas), "cublasCreate");
  check_cublas(cublasSetStream(impl_->cublas, impl_->stream), "cublasSetStream");
  check_cublas(cublasSetMathMode(impl_->cublas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
}

Context::~Context() {
  if (!impl_) return;
  if (impl_->cublas) cublasDestroy(impl_->cublas);
  if (impl_->cudnn) cudnnDestroy(impl_->cudnn);
  if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

void Context::synchronize() const {
  check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
}

}  // namespace nn
