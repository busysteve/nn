#pragma once

#include "nn/layer.hpp"

#include <cstdint>
#include <vector>

namespace nn {

class Sgd {
 public:
  Sgd(std::vector<NamedParam> params, float learning_rate, float momentum);
  Sgd(Sgd&&) noexcept = default;
  Sgd& operator=(Sgd&&) noexcept = default;
  Sgd(const Sgd&) = delete;
  Sgd& operator=(const Sgd&) = delete;

  void step();
  void set_learning_rate(float learning_rate) { lr_ = learning_rate; }
  float learning_rate() const { return lr_; }
  float momentum() const { return momentum_; }
  uint64_t step_count() const { return step_; }
  void set_step_count(uint64_t step) { step_ = step; }

  const std::vector<Tensor>& velocity() const { return velocity_; }
  std::vector<Tensor>& velocity() { return velocity_; }
  const std::vector<NamedParam>& params() const { return params_; }

 private:
  std::vector<NamedParam> params_;
  std::vector<Tensor> velocity_;
  float lr_ = 0.f;
  float momentum_ = 0.f;
  uint64_t step_ = 0;
};

}  // namespace nn
