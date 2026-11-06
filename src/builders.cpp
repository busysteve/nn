#include "nn/builders.hpp"

#include "nn/layers/batch_norm2d.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/flatten.hpp"
#include "nn/layers/global_avg_pool2d.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/max_pool2d.hpp"
#include "nn/layers/relu.hpp"
#include "nn/layers/residual_block.hpp"

#include <random>

namespace nn {
namespace {

std::string index_name(const std::string& prefix, size_t index) {
  return prefix + "." + std::to_string(index);
}

}  // namespace

Network build_mlp(Context& ctx, const MlpConfig& config, uint32_t seed) {
  validate(config);
  std::mt19937 rng(seed);
  std::vector<int> widths = config.hidden;
  widths.push_back(config.output_size);
  std::vector<std::unique_ptr<Layer>> layers;
  int in = config.input_size;
  for (size_t i = 0; i < widths.size(); ++i) {
    auto linear = std::make_unique<Linear>(ctx, in, widths[i], rng);
    linear->set_name(index_name("fc", i));
    layers.push_back(std::move(linear));
    if (i + 1 != widths.size()) layers.push_back(std::make_unique<ReLU>(ctx));
    in = widths[i];
  }
  return Network(ctx, Family::Mlp, std::move(layers), config);
}

Network build_conv(Context& ctx, const ConvConfig& config, uint32_t seed) {
  validate(config);
  std::mt19937 rng(seed);
  std::vector<std::unique_ptr<Layer>> layers;
  int channels = config.in_channels;
  for (size_t i = 0; i < config.convs.size(); ++i) {
    const ConvSpec& spec = config.convs[i];
    auto conv = std::make_unique<Conv2d>(ctx, channels, spec.out_channels, spec.kernel, spec.stride,
                                         spec.pad, true, rng);
    conv->set_name(index_name("conv", i));
    layers.push_back(std::move(conv));
    layers.push_back(std::make_unique<ReLU>(ctx));
    auto pool = std::make_unique<MaxPool2d>(ctx, config.pool.kernel, config.pool.stride);
    pool->set_name(index_name("pool", i));
    layers.push_back(std::move(pool));
    channels = spec.out_channels;
  }
  layers.push_back(std::make_unique<Flatten>());
  int features = channels;
  int height = config.in_height;
  int width = config.in_width;
  for (const ConvSpec& spec : config.convs) {
    height = conv_out_size(height, spec.kernel, spec.stride, spec.pad);
    width = conv_out_size(width, spec.kernel, spec.stride, spec.pad);
    height = conv_out_size(height, config.pool.kernel, config.pool.stride, 0);
    width = conv_out_size(width, config.pool.kernel, config.pool.stride, 0);
  }
  features = channels * height * width;
  for (size_t i = 0; i < config.fc.size(); ++i) {
    auto linear = std::make_unique<Linear>(ctx, features, config.fc[i], rng);
    linear->set_name(index_name("fc", i));
    layers.push_back(std::move(linear));
    if (i + 1 != config.fc.size()) layers.push_back(std::make_unique<ReLU>(ctx));
    features = config.fc[i];
  }
  return Network(ctx, Family::Conv, std::move(layers), config);
}

Network build_resnet(Context& ctx, const ResNetConfig& config, uint32_t seed) {
  validate(config);
  std::mt19937 rng(seed);
  std::vector<std::unique_ptr<Layer>> layers;
  auto stem = std::make_unique<Conv2d>(ctx, config.in_channels, config.stem_channels, 3, 1, 1, false, rng);
  stem->set_name("stem");
  layers.push_back(std::move(stem));
  auto stem_bn = std::make_unique<BatchNorm2d>(ctx, config.stem_channels);
  stem_bn->set_name("stem.bn");
  layers.push_back(std::move(stem_bn));
  layers.push_back(std::make_unique<ReLU>(ctx));

  int channels = config.stem_channels;
  for (size_t stage = 0; stage < config.blocks_per_stage.size(); ++stage) {
    const int width = config.stage_widths[stage];
    for (int block = 0; block < config.blocks_per_stage[stage]; ++block) {
      const int stride = (stage > 0 && block == 0) ? 2 : 1;
      auto residual = std::make_unique<ResidualBlock>(ctx, channels, width, stride, rng);
      residual->set_name("stages." + std::to_string(stage) + ".blocks." + std::to_string(block));
      layers.push_back(std::move(residual));
      channels = width;
    }
  }
  layers.push_back(std::make_unique<GlobalAvgPool2d>(ctx));
  layers.push_back(std::make_unique<Flatten>());
  auto head = std::make_unique<Linear>(ctx, channels, config.output_size, rng);
  head->set_name("head");
  layers.push_back(std::move(head));
  return Network(ctx, Family::ResNet, std::move(layers), config);
}

Network build_conv_resnet(Context& ctx, const ConvResNetConfig& config, uint32_t seed) {
  validate(config);
  std::mt19937 rng(seed);
  std::vector<std::unique_ptr<Layer>> layers;
  int channels = config.in_channels;
  int height = config.in_height;
  int width = config.in_width;

  for (size_t i = 0; i < config.convs.size(); ++i) {
    const ConvSpec& spec = config.convs[i];
    auto conv = std::make_unique<Conv2d>(ctx, channels, spec.out_channels, spec.kernel, spec.stride,
                                         spec.pad, true, rng);
    conv->set_name(index_name("conv", i));
    layers.push_back(std::move(conv));
    layers.push_back(std::make_unique<ReLU>(ctx));
    auto pool = std::make_unique<MaxPool2d>(ctx, config.pool.kernel, config.pool.stride);
    pool->set_name(index_name("pool", i));
    layers.push_back(std::move(pool));
    channels = spec.out_channels;
    height = conv_out_size(height, spec.kernel, spec.stride, spec.pad);
    width = conv_out_size(width, spec.kernel, spec.stride, spec.pad);
    height = conv_out_size(height, config.pool.kernel, config.pool.stride, 0);
    width = conv_out_size(width, config.pool.kernel, config.pool.stride, 0);
  }

  if (config.stem_channels > 0) {
    auto stem = std::make_unique<Conv2d>(ctx, channels, config.stem_channels, 3, 1, 1, false, rng);
    stem->set_name("stem");
    layers.push_back(std::move(stem));
    auto stem_bn = std::make_unique<BatchNorm2d>(ctx, config.stem_channels);
    stem_bn->set_name("stem.bn");
    layers.push_back(std::move(stem_bn));
    layers.push_back(std::make_unique<ReLU>(ctx));
    channels = config.stem_channels;
  }

  for (size_t stage = 0; stage < config.blocks_per_stage.size(); ++stage) {
    const int stage_width = config.stage_widths[stage];
    for (int block = 0; block < config.blocks_per_stage[stage]; ++block) {
      const int stride = (stage > 0 && block == 0) ? 2 : 1;
      auto residual = std::make_unique<ResidualBlock>(ctx, channels, stage_width, stride, rng);
      residual->set_name("stages." + std::to_string(stage) + ".blocks." + std::to_string(block));
      layers.push_back(std::move(residual));
      channels = stage_width;
    }
  }

  int features = channels;
  if (!config.blocks_per_stage.empty()) {
    layers.push_back(std::make_unique<GlobalAvgPool2d>(ctx));
  } else {
    features = channels * height * width;
  }
  layers.push_back(std::make_unique<Flatten>());
  for (size_t i = 0; i < config.fc.size(); ++i) {
    auto linear = std::make_unique<Linear>(ctx, features, config.fc[i], rng);
    linear->set_name(index_name("fc", i));
    layers.push_back(std::move(linear));
    if (i + 1 != config.fc.size()) layers.push_back(std::make_unique<ReLU>(ctx));
    features = config.fc[i];
  }
  return Network(ctx, Family::ConvResNet, std::move(layers), config);
}

}  // namespace nn
