#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/residual_block.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect_params(const nn::Network& net,
                   const std::vector<std::pair<std::string, std::vector<int>>>& expected) {
  const std::vector<nn::NamedParam> params = net.parameters();
  if (params.size() != expected.size()) {
    std::fprintf(stderr, "parameter count %zu, expected %zu\n", params.size(), expected.size());
    CHECK(false);
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (params[i].name != expected[i].first || params[i].value->shape() != expected[i].second) {
      std::fprintf(stderr, "param %zu got %s rank %d, expected %s\n", i, params[i].name.c_str(),
                   params[i].value->rank(), expected[i].first.c_str());
      CHECK(params[i].name == expected[i].first);
      CHECK(params[i].value->shape() == expected[i].second);
    }
  }
}

bool has_param(const nn::Network& net, const std::string& name) {
  for (const nn::NamedParam& param : net.parameters()) {
    if (param.name == name) return true;
  }
  return false;
}

nn::Tensor* find_param(nn::Network& net, const std::string& name) {
  for (nn::NamedParam& param : net.parameters()) {
    if (param.name == name) return param.value;
  }
  return nullptr;
}

void check_io(nn::Context& ctx, nn::Network& net, const std::vector<int>& input_shape,
              const std::vector<int>& output_shape) {
  net.zero_grad();
  nn::Tensor input(ctx, input_shape);
  input.fill(0.1f);
  const nn::Tensor& output = net.forward(input);
  CHECK(output.shape() == output_shape);
  nn::Tensor grad_output(ctx, output.shape());
  grad_output.fill(1.f);
  const nn::Tensor& grad_input = net.backward(grad_output);
  CHECK(grad_input.shape() == input_shape);
}

void check_block(nn::Context& ctx, int in_channels, int out_channels, int stride,
                 const std::vector<int>& input_shape, const std::vector<int>& output_shape,
                 bool expect_proj) {
  std::mt19937 rng(1);
  nn::ResidualBlock block(ctx, in_channels, out_channels, stride, rng);
  block.zero_grad();
  nn::Tensor input(ctx, input_shape);
  input.fill(0.1f);
  const nn::Tensor& output = block.forward(input);
  CHECK(output.shape() == output_shape);
  nn::Tensor grad_output(ctx, output.shape());
  grad_output.fill(1.f);
  const nn::Tensor& grad_input = block.backward(grad_output);
  CHECK(grad_input.shape() == input_shape);
  bool found = false;
  for (const nn::NamedParam& param : block.parameters()) {
    if (param.name == "proj.weight") found = true;
  }
  CHECK(found == expect_proj);
}

}  // namespace

int test_shapes() {
  nn::Context ctx;
  std::mt19937 rng(1);

  {
    nn::Linear linear(ctx, 2, 2, rng);
    std::vector<nn::NamedParam> params = linear.parameters();
    CHECK(params.size() == 2);
    params[0].value->copy_from_host(std::vector<float>{3.f, 4.f, 5.f, 6.f});
    params[1].value->copy_from_host(std::vector<float>{0.f, 0.f});
    nn::Tensor input(ctx, {1, 2});
    input.copy_from_host(std::vector<float>{1.f, 2.f});
    const nn::Tensor& output = linear.forward(input);
    const std::vector<float> got = output.to_host();
    CHECK(got.size() == 2);
    CHECK(got[0] == 11.f);
    CHECK(got[1] == 17.f);

    nn::Tensor grad_output(ctx, {1, 2});
    grad_output.copy_from_host(std::vector<float>{1.5f, -0.25f});
    const std::vector<float> before = grad_output.to_host();
    linear.backward(grad_output);
    const std::vector<float> after = grad_output.to_host();
    CHECK(before.size() == after.size());
    CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0);
  }

  {
    nn::MlpConfig config;
    config.input_size = 784;
    config.hidden = {256, 128};
    config.output_size = 10;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {256, 784}},
                        {"fc.0.bias", {256}},
                        {"fc.1.weight", {128, 256}},
                        {"fc.1.bias", {128}},
                        {"fc.2.weight", {10, 128}},
                        {"fc.2.bias", {10}}});
    check_io(ctx, net, {3, 784}, {3, 10});
  }

  {
    nn::MlpConfig config;
    config.input_size = 16;
    config.hidden = {};
    config.output_size = 4;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {4, 16}}, {"fc.0.bias", {4}}});
    check_io(ctx, net, {2, 16}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {7, 6, 5};
    config.output_size = 4;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {7, 8}},
                        {"fc.0.bias", {7}},
                        {"fc.1.weight", {6, 7}},
                        {"fc.1.bias", {6}},
                        {"fc.2.weight", {5, 6}},
                        {"fc.2.bias", {5}},
                        {"fc.3.weight", {4, 5}},
                        {"fc.3.bias", {4}}});
    check_io(ctx, net, {1, 8}, {1, 4});
  }

  CHECK(nn::conv_out_size(28, 5, 1, 2) == 28);
  CHECK(nn::conv_out_size(28, 2, 2, 0) == 14);
  CHECK(nn::conv_out_size(14, 5, 1, 2) == 14);
  CHECK(nn::conv_out_size(14, 2, 2, 0) == 7);
  CHECK(nn::conv_out_size(7, 2, 2, 0) == 3);
  CHECK(nn::conv_out_size(28, 3, 1, 1) == 28);
  CHECK(nn::conv_out_size(28, 3, 2, 1) == 14);
  CHECK(nn::conv_out_size(14, 3, 2, 1) == 7);

  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{32, 5, 1, 2}, nn::ConvSpec{64, 5, 1, 2}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {128, 10};
    nn::Network net = nn::build_conv(ctx, config, 1);
    expect_params(net, {{"conv.0.weight", {32, 1, 5, 5}},
                        {"conv.0.bias", {32}},
                        {"conv.1.weight", {64, 32, 5, 5}},
                        {"conv.1.bias", {64}},
                        {"fc.0.weight", {128, 3136}},
                        {"fc.0.bias", {128}},
                        {"fc.1.weight", {10, 128}},
                        {"fc.1.bias", {10}}});
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.convs = {nn::ConvSpec{8, 3, 1, 1}, nn::ConvSpec{8, 3, 1, 1}, nn::ConvSpec{16, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {10};
    nn::Network net = nn::build_conv(ctx, config, 1);
    nn::Tensor* weight = find_param(net, "fc.0.weight");
    CHECK(weight != nullptr);
    CHECK(weight->shape() == std::vector<int>({10, 144}));
    nn::Tensor* conv2 = find_param(net, "conv.2.weight");
    CHECK(conv2 != nullptr);
    CHECK(conv2->shape() == std::vector<int>({16, 8, 3, 3}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.stem_channels = 16;
    config.blocks_per_stage = {2, 2, 2};
    config.stage_widths = {16, 32, 64};
    config.output_size = 10;
    nn::Network net = nn::build_resnet(ctx, config, 1);
    CHECK(find_param(net, "stem.weight")->shape() == std::vector<int>({16, 1, 3, 3}));
    CHECK(find_param(net, "head.weight")->shape() == std::vector<int>({10, 64}));
    CHECK(find_param(net, "head.bias")->shape() == std::vector<int>({10}));
    CHECK(!has_param(net, "stages.0.blocks.0.proj.weight"));
    CHECK(!has_param(net, "stages.0.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.1.blocks.0.proj.weight")->shape() == std::vector<int>({32, 16, 1, 1}));
    CHECK(!has_param(net, "stages.1.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.2.blocks.0.proj.weight")->shape() == std::vector<int>({64, 32, 1, 1}));
    CHECK(!has_param(net, "stages.2.blocks.1.proj.weight"));
    CHECK(find_param(net, "stages.0.blocks.0.conv1.weight")->shape() == std::vector<int>({16, 16, 3, 3}));
    CHECK(find_param(net, "stages.1.blocks.0.conv1.weight")->shape() == std::vector<int>({32, 16, 3, 3}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 10});
  }

  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 28;
    config.in_width = 28;
    config.stem_channels = 8;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {8, 16};
    config.output_size = 5;
    nn::Network net = nn::build_resnet(ctx, config, 1);
    CHECK(find_param(net, "stem.weight")->shape() == std::vector<int>({8, 1, 3, 3}));
    CHECK(find_param(net, "head.weight")->shape() == std::vector<int>({5, 16}));
    CHECK(!has_param(net, "stages.0.blocks.0.proj.weight"));
    CHECK(find_param(net, "stages.1.blocks.0.proj.weight")->shape() == std::vector<int>({16, 8, 1, 1}));
    check_io(ctx, net, {2, 1, 28, 28}, {2, 5});

    std::mt19937 stem_rng(1);
    nn::Conv2d stem(ctx, 1, 8, 3, 1, 1, false, stem_rng);
    nn::Tensor image(ctx, {2, 1, 28, 28});
    image.fill(0.1f);
    CHECK(stem.forward(image).shape() == std::vector<int>({2, 8, 28, 28}));
    nn::ResidualBlock stage0(ctx, 8, 8, 1, stem_rng);
    nn::Tensor map(ctx, {2, 8, 28, 28});
    map.fill(0.1f);
    CHECK(stage0.forward(map).shape() == std::vector<int>({2, 8, 28, 28}));
    nn::ResidualBlock stage1(ctx, 8, 16, 2, stem_rng);
    CHECK(stage1.forward(map).shape() == std::vector<int>({2, 16, 14, 14}));
  }

  check_block(ctx, 4, 4, 1, {2, 4, 8, 8}, {2, 4, 8, 8}, false);
  check_block(ctx, 4, 8, 1, {2, 4, 8, 8}, {2, 8, 8, 8}, true);
  check_block(ctx, 4, 8, 2, {2, 4, 8, 8}, {2, 8, 4, 4}, true);

  std::printf("shapes ok\n");
  return 0;
}
