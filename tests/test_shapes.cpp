#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/conv_resnet.hpp"
#include "nn/error.hpp"
#include "nn/index_tensor.hpp"
#include "nn/init.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/embedding.hpp"
#include "nn/layers/gelu.hpp"
#include "nn/layers/last_step.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/residual_block.hpp"
#include "nn/layers/rnn.hpp"
#include "nn/layers/swiglu.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
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

void expect_error(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const nn::Error& error) {
    threw = true;
    if (message != error.what()) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

void expect_invalid(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const std::invalid_argument& error) {
    threw = true;
    if (message != error.what()) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

void expect_close(const std::vector<float>& got, const std::vector<float>& want, const char* name) {
  CHECK(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double scale = std::max(1.0, std::max(std::abs(static_cast<double>(got[i])),
                                                std::abs(static_cast<double>(want[i]))));
    if (diff / scale <= 1e-5 || diff <= 1e-5) continue;
    std::fprintf(stderr, "%s[%zu] got %.8g want %.8g\n", name, i, static_cast<double>(got[i]),
                 static_cast<double>(want[i]));
    CHECK(false);
  }
}

float gelu_ref(float x) {
  return 0.5f * x * (1.f + std::erff(x * 0.7071067811865476f));
}

float silu_ref(float x) {
  const float sigmoid = (x >= 0.f) ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x));
  return x * sigmoid;
}

std::vector<float> swiglu_ref(const std::vector<float>& input, int rows, int inner) {
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(inner));
  for (int row = 0; row < rows; ++row) {
    const float* gate = input.data() + static_cast<size_t>(row) * static_cast<size_t>(inner) * 2;
    const float* value = gate + inner;
    for (int col = 0; col < inner; ++col) {
      output[static_cast<size_t>(row) * static_cast<size_t>(inner) + static_cast<size_t>(col)] =
          silu_ref(gate[col]) * value[col];
    }
  }
  return output;
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
    nn::GELU layer(ctx);
    const std::vector<float> host = {0.f, 1.f, -1.f, 0.5f, -0.25f, 2.f};
    nn::Tensor input(ctx, {2, 3});
    input.copy_from_host(host);
    std::vector<float> want(host.size());
    for (size_t i = 0; i < host.size(); ++i) want[i] = gelu_ref(host[i]);
    expect_close(layer.forward(input).to_host(), want, "gelu");
    nn::Tensor grad(ctx, {2, 3});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());
    nn::Tensor rank4(ctx, {1, 2, 2, 2});
    rank4.fill(0.25f);
    CHECK(layer.forward(rank4).shape() == rank4.shape());
    nn::Tensor bad_rank(ctx, {2, 3, 4});
    expect_error([&] { layer.forward(bad_rank); }, "GELU rank");
    nn::GELU fresh(ctx);
    nn::Tensor grad_rank(ctx, {2, 2});
    expect_error([&] { fresh.backward(grad_rank); }, "GELU backward shape");
  }

  {
    nn::SwiGLU layer(ctx);
    const std::vector<float> host = {1.f, -0.5f, 0.25f, -1.f, 0.5f, 2.f};
    nn::Tensor input(ctx, {1, 6});
    input.copy_from_host(host);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 3}));
    expect_close(output.to_host(), swiglu_ref(host, 1, 3), "swiglu");
    nn::Tensor grad(ctx, {1, 3});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());

    const std::vector<float> nchw = {0.5f, -0.25f, 1.5f, -1.f, 0.75f, 0.125f, -0.5f, 2.f};
    nn::Tensor image(ctx, {1, 4, 1, 2});
    image.copy_from_host(nchw);
    const nn::Tensor& pooled = layer.forward(image);
    CHECK(pooled.shape() == std::vector<int>({1, 2, 1, 2}));
    expect_close(pooled.to_host(), swiglu_ref(nchw, 1, 4), "swiglu-nchw");

    nn::Tensor odd(ctx, {2, 3});
    expect_error([&] { layer.forward(odd); }, "SwiGLU size");
    nn::Tensor bad_rank(ctx, {2, 2, 2});
    expect_error([&] { layer.forward(bad_rank); }, "SwiGLU rank");
    nn::Tensor bad_grad(ctx, {1, 6});
    expect_error([&] { layer.backward(bad_grad); }, "SwiGLU backward shape");
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

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {7, 6};
    config.output_size = 4;
    config.activation = nn::Activation::Gelu;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {7, 8}},
                        {"fc.0.bias", {7}},
                        {"fc.1.weight", {6, 7}},
                        {"fc.1.bias", {6}},
                        {"fc.2.weight", {4, 6}},
                        {"fc.2.bias", {4}}});
    check_io(ctx, net, {2, 8}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6, 5};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {12, 8}},
                        {"fc.0.bias", {12}},
                        {"fc.1.weight", {10, 6}},
                        {"fc.1.bias", {10}},
                        {"fc.2.weight", {4, 5}},
                        {"fc.2.bias", {4}}});
    check_io(ctx, net, {2, 8}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 16;
    config.hidden = {};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    nn::Network net = nn::build_mlp(ctx, config, 1);
    expect_params(net, {{"fc.0.weight", {4, 16}}, {"fc.0.bias", {4}}});
    check_io(ctx, net, {2, 16}, {2, 4});
  }

  {
    nn::MlpConfig config;
    config.input_size = 4;
    config.hidden = {4};
    config.output_size = 2;
    config.activation = static_cast<nn::Activation>(0);
    expect_invalid([&] { nn::validate(config); }, "activation");
    config.activation = nn::Activation::SwiGLU;
    config.hidden = {1 << 30};
    expect_invalid([&] { nn::validate(config); }, "hidden");
  }

  {
    nn::LastStep layer;
    const std::vector<float> host = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    nn::Tensor input(ctx, {2, 3, 2});
    input.copy_from_host(host);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({2, 2}));
    expect_close(output.to_host(), {4.f, 5.f, 10.f, 11.f}, "last-step");
    nn::Tensor grad(ctx, {2, 2});
    grad.copy_from_host(std::vector<float>{1.f, 2.f, 3.f, 4.f});
    const std::vector<float> got = layer.backward(grad).to_host();
    expect_close(got, {0.f, 0.f, 0.f, 0.f, 1.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 4.f}, "last-step-grad");
    nn::Tensor bad_rank(ctx, {2, 3});
    expect_error([&] { layer.forward(bad_rank); }, "LastStep input rank");
    nn::LastStep fresh;
    expect_error([&] { fresh.backward(grad); }, "LastStep backward without forward");
  }

  {
    nn::RNN layer(ctx, 1, 1, rng);
    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 4);
    params[0].value->copy_from_host(std::vector<float>{1.f});
    params[1].value->copy_from_host(std::vector<float>{1.f});
    params[2].value->fill(0.f);
    params[3].value->fill(0.f);
    nn::Tensor input(ctx, {1, 2, 1});
    input.copy_from_host(std::vector<float>{0.5f, 0.25f});
    const float h0 = std::tanh(0.5f);
    const float h1 = std::tanh(0.25f + h0);
    const nn::Tensor& output = layer.forward(input);
    CHECK(output.shape() == std::vector<int>({1, 2, 1}));
    expect_close(output.to_host(), {h0, h1}, "rnn");
    nn::Tensor grad(ctx, {1, 2, 1});
    grad.fill(1.f);
    CHECK(layer.backward(grad).shape() == input.shape());
    nn::Tensor bad_rank(ctx, {1, 1});
    expect_error([&] { layer.forward(bad_rank); }, "RNN input shape");
    expect_error([&] { nn::RNN bad(ctx, 0, 1, rng); }, "RNN size");
  }

  {
    nn::RnnConfig config;
    config.input_size = 3;
    config.hidden = {4, 6};
    config.output_size = 2;
    nn::Network net = nn::build_rnn(ctx, config, 1);
    expect_params(net, {{"rnn.0.weight_ih", {4, 3}},
                        {"rnn.0.weight_hh", {4, 4}},
                        {"rnn.0.bias_ih", {4}},
                        {"rnn.0.bias_hh", {4}},
                        {"rnn.1.weight_ih", {6, 4}},
                        {"rnn.1.weight_hh", {6, 6}},
                        {"rnn.1.bias_ih", {6}},
                        {"rnn.1.bias_hh", {6}},
                        {"head.weight", {2, 6}},
                        {"head.bias", {2}}});
    check_io(ctx, net, {2, 5, 3}, {2, 2});
    CHECK(net.family() == nn::Family::Rnn);
  }

  {
    nn::RnnConfig config;
    config.input_size = 4;
    config.output_size = 2;
    expect_invalid([&] { nn::validate(config); }, "hidden");
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

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{4, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    CHECK(model.family() == nn::Family::ConvResNet);
    CHECK(model.training());
    CHECK(model.config().stem_channels == 4);
    CHECK(model.parameters().size() == model.network().parameters().size());
    expect_params(model.network(), {{"conv.0.weight", {4, 1, 3, 3}},
                                    {"conv.0.bias", {4}},
                                    {"stem.weight", {4, 4, 3, 3}},
                                    {"stem.bn.weight", {4}},
                                    {"stem.bn.bias", {4}},
                                    {"stages.0.blocks.0.conv1.weight", {4, 4, 3, 3}},
                                    {"stages.0.blocks.0.bn1.weight", {4}},
                                    {"stages.0.blocks.0.bn1.bias", {4}},
                                    {"stages.0.blocks.0.conv2.weight", {4, 4, 3, 3}},
                                    {"stages.0.blocks.0.bn2.weight", {4}},
                                    {"stages.0.blocks.0.bn2.bias", {4}},
                                    {"stages.1.blocks.0.conv1.weight", {8, 4, 3, 3}},
                                    {"stages.1.blocks.0.bn1.weight", {8}},
                                    {"stages.1.blocks.0.bn1.bias", {8}},
                                    {"stages.1.blocks.0.conv2.weight", {8, 8, 3, 3}},
                                    {"stages.1.blocks.0.bn2.weight", {8}},
                                    {"stages.1.blocks.0.bn2.bias", {8}},
                                    {"stages.1.blocks.0.proj.weight", {8, 4, 1, 1}},
                                    {"stages.1.blocks.0.proj_bn.weight", {8}},
                                    {"stages.1.blocks.0.proj_bn.bias", {8}},
                                    {"fc.0.weight", {4, 8}},
                                    {"fc.0.bias", {4}}});
    CHECK(!has_param(model.network(), "stages.0.blocks.0.proj.weight"));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{4, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    expect_params(model.network(), {{"conv.0.weight", {4, 1, 3, 3}},
                                    {"conv.0.bias", {4}},
                                    {"fc.0.weight", {4, 64}},
                                    {"fc.0.bias", {4}}});
    CHECK(!has_param(model.network(), "stem.weight"));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::ConvResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.fc = {4};
    nn::ConvResNet model(ctx, config, 1);
    CHECK(find_param(model.network(), "stem.weight")->shape() == std::vector<int>({4, 1, 3, 3}));
    CHECK(!has_param(model.network(), "conv.0.weight"));
    CHECK(!has_param(model.network(), "stages.0.blocks.0.proj.weight"));
    CHECK(find_param(model.network(), "stages.1.blocks.0.proj.weight")->shape() ==
          std::vector<int>({8, 4, 1, 1}));
    CHECK(find_param(model.network(), "fc.0.weight")->shape() == std::vector<int>({4, 8}));
    check_io(ctx, model.network(), {2, 1, 8, 8}, {2, 4});
  }

  {
    nn::MlpConfig mlp;
    mlp.input_size = 4;
    mlp.output_size = 2;
    bool threw = false;
    try {
      nn::ConvResNet model(nn::build_mlp(ctx, mlp, 1));
      (void)model;
    } catch (const nn::Error& error) {
      threw = true;
      CHECK(std::string(error.what()) == "ConvResNet family");
    }
    CHECK(threw);
  }

  {
    expect_error([&] { nn::IndexTensor(ctx, {}); }, "index tensor rank out of range");
    expect_error([&] { nn::IndexTensor(ctx, {1, 1, 1}); }, "index tensor rank out of range");
    expect_error([&] { nn::IndexTensor(ctx, {0}); }, "index tensor dimension out of range");
    expect_error([&] { nn::IndexTensor(ctx, {2, 0}); }, "index tensor dimension out of range");
    expect_error([&] { nn::IndexTensor(ctx, {std::numeric_limits<int>::max(), 2}); },
                 "index tensor numel out of range");

    nn::IndexTensor ids(ctx, {2, 2});
    CHECK(ids.data() != nullptr);
    CHECK(ids.numel() == 4);
    CHECK(ids.shape() == std::vector<int>({2, 2}));
    CHECK(&ids.context() == &ctx);
    const int64_t one = 0;
    expect_error([&] { ids.copy_from_host(&one, 1); }, "index tensor host copy numel mismatch");

    nn::Tensor sample(ctx, {4});
    expect_error([&] { nn::normal_(sample, 0.f, 0.f, rng); }, "normal_ std");
    expect_error([&] { nn::normal_(sample, 0.f, -0.1f, rng); }, "normal_ std");
    expect_error([&] { nn::normal_(sample, std::numeric_limits<float>::infinity(), 0.02f, rng); },
                 "normal_ mean");
    std::mt19937 draw_rng(5);
    std::mt19937 ref_rng(5);
    nn::normal_(sample, 0.25f, 0.02f, draw_rng);
    std::normal_distribution<float> dist(0.25f, 0.02f);
    std::vector<float> drawn(4);
    for (float& value : drawn) value = dist(ref_rng);
    expect_close(sample.to_host(), drawn, "normal_");

    expect_error([&] { nn::Embedding(ctx, 0, 2, rng); }, "Embedding size");
    expect_error([&] { nn::Embedding(ctx, 2, -1, rng); }, "Embedding size");

    nn::Embedding layer(ctx, 3, 2, rng);
    nn::Tensor float_input(ctx, {2, 2});
    expect_error([&] { layer.forward(float_input); }, "Embedding requires indices");

    std::vector<nn::NamedParam> params = layer.parameters();
    CHECK(params.size() == 1);
    CHECK(params[0].name == "weight");
    CHECK(params[0].value->shape() == std::vector<int>({3, 2}));
    CHECK(params[0].grad->shape() == params[0].value->shape());
    params[0].value->copy_from_host(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    const std::vector<int64_t> host_ids = {2, 0, 1, 0};
    ids.copy_from_host(host_ids.data(), static_cast<int64_t>(host_ids.size()));
    const nn::Tensor& output = layer.forward_indices(ids);
    CHECK(output.shape() == std::vector<int>({2, 2, 2}));
    expect_close(output.to_host(), {5.f, 6.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f}, "embedding");

    nn::IndexTensor bad(ctx, {2});
    const std::vector<int64_t> oob = {0, 3};
    bad.copy_from_host(oob.data(), static_cast<int64_t>(oob.size()));
    expect_error([&] { layer.forward_indices(bad); }, "index out of range");
    const std::vector<int64_t> neg = {-1, 1};
    bad.copy_from_host(neg.data(), static_cast<int64_t>(neg.size()));
    expect_error([&] { layer.forward_indices(bad); }, "index out of range");

    // A rejected index must leave the ids captured by the last successful forward.
    layer.zero_grad();
    expect_close(params[0].grad->to_host(), std::vector<float>(6, 0.f), "embedding zero");
    nn::Tensor grad(ctx, {2, 2, 2});
    grad.copy_from_host(std::vector<float>{1.f, 10.f, 2.f, 20.f, 3.f, 30.f, 4.f, 40.f});
    const std::vector<float> before = grad.to_host();
    const nn::Tensor& back = layer.backward(grad);
    CHECK(back.shape() == grad.shape());
    expect_close(back.to_host(), std::vector<float>(static_cast<size_t>(grad.numel()), 0.f),
                 "embedding backward return");
    CHECK(std::memcmp(before.data(), grad.to_host().data(), before.size() * sizeof(float)) == 0);
    expect_close(params[0].grad->to_host(), {6.f, 60.f, 3.f, 30.f, 1.f, 10.f}, "embedding grad");
    layer.backward(grad);
    expect_close(params[0].grad->to_host(), {12.f, 120.f, 6.f, 60.f, 2.f, 20.f},
                 "embedding grad accum");

    nn::Tensor bad_grad(ctx, {2, 2});
    expect_error([&] { layer.backward(bad_grad); }, "Embedding backward shape");
    nn::Embedding fresh(ctx, 2, 2, rng);
    expect_error([&] { fresh.backward(bad_grad); }, "Embedding backward without forward");

    nn::IndexTensor rows(ctx, {3});
    const std::vector<int64_t> row_ids = {1, 2, 0};
    rows.copy_from_host(row_ids.data(), static_cast<int64_t>(row_ids.size()));
    const nn::Tensor& row_out = layer.forward_indices(rows);
    CHECK(row_out.shape() == std::vector<int>({3, 2}));
    expect_close(row_out.to_host(), {3.f, 4.f, 5.f, 6.f, 1.f, 2.f}, "embedding rows");

    nn::Context other;
    nn::IndexTensor foreign(other, {1});
    const int64_t zero = 0;
    foreign.copy_from_host(&zero, 1);
    expect_error([&] { layer.forward_indices(foreign); }, "Embedding context");
  }

  std::printf("shapes ok\n");
  return 0;
}
