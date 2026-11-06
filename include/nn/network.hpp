#pragma once

#include "nn/checkpoint.hpp"
#include "nn/config.hpp"
#include "nn/layer.hpp"
#include "nn/optim.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace nn {

enum class Family : uint32_t { Mlp = 1, Conv = 2, ResNet = 3, ConvResNet = 4 };

class Network {
 public:
  Network(Context& ctx, Family family, std::vector<std::unique_ptr<Layer>> layers,
          std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig> config);
  Network(Network&&) noexcept = default;
  Network& operator=(Network&&) noexcept = default;
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  const Tensor& forward(const Tensor& input);
  const Tensor& backward(const Tensor& grad_output);
  void zero_grad();
  void train();
  void eval();
  bool training() const { return training_; }
  Family family() const { return family_; }
  const std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig>& config() const {
    return config_;
  }

  std::vector<NamedParam> parameters() const;
  std::vector<NamedBuffer> buffers() const;

  void save(const std::filesystem::path& path, const Sgd* optimizer) const;

  struct Loaded;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);

 private:
  Context* ctx_ = nullptr;
  Family family_ = Family::Mlp;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig> config_;
  bool training_ = true;
};

struct Network::Loaded {
  Network network;
  std::optional<Sgd> optimizer;
  uint64_t step = 0;
};

}  // namespace nn
