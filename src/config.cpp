#include "nn/config.hpp"

#include <stdexcept>
#include <string>

namespace nn {
namespace {

[[noreturn]] void bad(const char* field) { throw std::invalid_argument(field); }

void require_positive(int value, const char* field) {
  if (value < 1) bad(field);
}

}  // namespace

int conv_out_size(int in, int kernel, int stride, int pad) {
  if (kernel < 1) bad("kernel");
  if (stride < 1) bad("stride");
  if (pad < 0) bad("pad");
  const int out = (in + 2 * pad - kernel) / stride + 1;
  if (out < 1) bad("spatial");
  return out;
}

void validate(const MlpConfig& config) {
  require_positive(config.input_size, "input_size");
  require_positive(config.output_size, "output_size");
  if (config.activation != Activation::Relu) bad("activation");
  for (int width : config.hidden) require_positive(width, "hidden");
}

void validate(const ConvConfig& config) {
  require_positive(config.in_channels, "in_channels");
  require_positive(config.in_height, "in_height");
  require_positive(config.in_width, "in_width");
  if (config.convs.empty()) bad("convs");
  if (config.fc.empty()) bad("fc");
  int height = config.in_height;
  int width = config.in_width;
  for (const ConvSpec& spec : config.convs) {
    require_positive(spec.out_channels, "out_channels");
    height = conv_out_size(height, spec.kernel, spec.stride, spec.pad);
    width = conv_out_size(width, spec.kernel, spec.stride, spec.pad);
    height = conv_out_size(height, config.pool.kernel, config.pool.stride, 0);
    width = conv_out_size(width, config.pool.kernel, config.pool.stride, 0);
  }
  for (int units : config.fc) require_positive(units, "fc");
}

void validate(const ResNetConfig& config) {
  require_positive(config.in_channels, "in_channels");
  require_positive(config.in_height, "in_height");
  require_positive(config.in_width, "in_width");
  require_positive(config.stem_channels, "stem_channels");
  require_positive(config.output_size, "output_size");
  if (config.blocks_per_stage.empty() ||
      config.blocks_per_stage.size() != config.stage_widths.size()) {
    bad("stage_widths");
  }
  int height = conv_out_size(config.in_height, 3, 1, 1);
  int width = conv_out_size(config.in_width, 3, 1, 1);
  for (size_t stage = 0; stage < config.blocks_per_stage.size(); ++stage) {
    require_positive(config.stage_widths[stage], "stage_widths");
    if (config.blocks_per_stage[stage] < 1) bad("blocks_per_stage");
    const int first_stride = stage == 0 ? 1 : 2;
    height = conv_out_size(height, 3, first_stride, 1);
    width = conv_out_size(width, 3, first_stride, 1);
  }
}

}  // namespace nn
