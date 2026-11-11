#include "context_impl.hpp"
#include "check.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace nn {
namespace {

std::string trimmed(std::string text) {
  auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
  while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
  return text;
}

bool is_auto(const std::string& text) {
  if (text.size() != 4) return false;
  return std::tolower(static_cast<unsigned char>(text[0])) == 'a' &&
         std::tolower(static_cast<unsigned char>(text[1])) == 'u' &&
         std::tolower(static_cast<unsigned char>(text[2])) == 't' &&
         std::tolower(static_cast<unsigned char>(text[3])) == 'o';
}

bool parse_index(const std::string& text, int& index) {
  if (text.empty()) return false;
  int value = 0;
  for (char ch : text) {
    if (ch < '0' || ch > '9') return false;
    const int digit = ch - '0';
    if (value > (2147483647 - digit) / 10) return false;
    value = value * 10 + digit;
  }
  index = value;
  return true;
}

}  // namespace

void Context::open_handles() {
  check_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
  check_cudnn(cudnnCreate(&impl_->cudnn), "cudnnCreate");
  check_cudnn(cudnnSetStream(impl_->cudnn, impl_->stream), "cudnnSetStream");
  check_cublas(cublasCreate(&impl_->cublas), "cublasCreate");
  check_cublas(cublasSetStream(impl_->cublas, impl_->stream), "cublasSetStream");
  check_cublas(cublasSetMathMode(impl_->cublas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
}

int device_count() {
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  return count;
}

std::string device_name(int device) {
  if (device < 0 || device >= device_count()) throw Error("device");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return prop.name;
}

std::vector<DeviceInfo> detect_devices() {
  const int count = device_count();
  std::vector<DeviceInfo> found;
  found.reserve(static_cast<size_t>(count));
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    found.push_back(DeviceInfo{device, prop.name});
  }
  return found;
}

int select_device(const std::string& choice) {
  const std::vector<DeviceInfo> found = detect_devices();
  if (found.empty()) throw Error("device");
  const std::string text = trimmed(choice);
  if (is_auto(text)) return found.front().index;
  int index = 0;
  if (!parse_index(text, index) || index >= static_cast<int>(found.size())) throw Error("device");
  return index;
}

Context::Context() : impl_(std::make_unique<Impl>()) { open_handles(); }

Context::Context(int device) : impl_(std::make_unique<Impl>()) {
  if (device < 0 || device >= device_count()) throw Error("device");
  check_cuda(cudaSetDevice(device), "cudaSetDevice");
  open_handles();
}

Context::Context(const std::string& choice) : impl_(std::make_unique<Impl>()) {
  check_cuda(cudaSetDevice(select_device(choice)), "cudaSetDevice");
  open_handles();
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
