#include "nn/init.hpp"

#include "nn/error.hpp"

#include <cmath>
#include <vector>

namespace nn {

void kaiming_normal(Tensor& weight, int fan_in, std::mt19937& rng) {
  if (fan_in < 1) throw Error("kaiming fan_in");
  const float stddev = std::sqrt(2.f / static_cast<float>(fan_in));
  std::normal_distribution<float> dist(0.f, stddev);
  std::vector<float> host(static_cast<size_t>(weight.numel()));
  for (float& value : host) value = dist(rng);
  weight.copy_from_host(host);
  weight.context().synchronize();
}

void normal_(Tensor& tensor, float mean, float stddev, std::mt19937& rng) {
  if (!(stddev > 0.f) || !std::isfinite(stddev)) throw Error("normal_ std");
  if (!std::isfinite(mean)) throw Error("normal_ mean");
  std::normal_distribution<float> dist(mean, stddev);
  std::vector<float> host(static_cast<size_t>(tensor.numel()));
  for (float& value : host) value = dist(rng);
  tensor.copy_from_host(host);
  tensor.context().synchronize();
}

}  // namespace nn
