#pragma once

#include "cudnn_api.hpp"
#include "nn/context.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace nn {

struct Context::Impl {
  cudaStream_t stream = nullptr;
  cudnnHandle_t cudnn = nullptr;
  cublasHandle_t cublas = nullptr;
};

struct ContextAccess {
  static Context::Impl& get(Context& ctx) { return *ctx.impl_; }
  static const Context::Impl& get(const Context& ctx) { return *ctx.impl_; }
};

inline cudaStream_t stream_of(Context& ctx) { return ContextAccess::get(ctx).stream; }
inline cudnnHandle_t cudnn_of(Context& ctx) { return ContextAccess::get(ctx).cudnn; }
inline cublasHandle_t cublas_of(Context& ctx) { return ContextAccess::get(ctx).cublas; }

}  // namespace nn
