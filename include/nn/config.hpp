#pragma once

#include <cstdint>
#include <vector>

namespace nn {

enum class Activation : uint32_t { Relu = 1 };

struct MlpConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
  Activation activation = Activation::Relu;
};

struct ConvSpec {
  int out_channels = 0;
  int kernel = 0;
  int stride = 0;
  int pad = 0;
};

struct PoolSpec {
  int kernel = 0;
  int stride = 0;
};

struct ConvConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  std::vector<int> fc;
};

struct ResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  int output_size = 0;
};

int conv_out_size(int in, int kernel, int stride, int pad);

void validate(const MlpConfig& config);
void validate(const ConvConfig& config);
void validate(const ResNetConfig& config);

}  // namespace nn
