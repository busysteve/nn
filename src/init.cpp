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

}  // namespace nn
