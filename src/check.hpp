#pragma once

#include "cudnn_api.hpp"
#include "nn/error.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace nn {

void check_cuda(cudaError_t status, const char* call);
void check_cudnn(cudnnStatus_t status, const char* call);
void check_cublas(cublasStatus_t status, const char* call);

}  // namespace nn
