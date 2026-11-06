#include "check.hpp"

#include "nn/layers/batch_norm2d.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/layers/gelu.hpp"
#include "nn/layers/last_step.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/max_pool2d.hpp"
#include "nn/layers/multi_head_attention.hpp"
#include "nn/layers/relu.hpp"
#include "nn/layers/residual_block.hpp"
#include "nn/layers/rnn.hpp"
#include "nn/layers/swiglu.hpp"
#include "nn/loss.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kEps = 1e-3;

void check_close(double analytic, double numeric, double rel, double abs_floor, const char* name,
                 size_t index) {
  const double diff = std::abs(analytic - numeric);
  const double scale = std::max(1.0, std::max(std::abs(analytic), std::abs(numeric)));
  if (diff / scale <= rel || diff <= abs_floor) return;
  std::fprintf(stderr, "grad mismatch %s[%zu] analytic=%.8g numeric=%.8g diff=%.8g rel=%.4g\n", name,
               index, analytic, numeric, diff, diff / scale);
  std::exit(1);
}

double squared_loss(const nn::Tensor& output) {
  const std::vector<float> values = output.to_host();
  double sum = 0.0;
  for (float value : values) sum += 0.5 * static_cast<double>(value) * static_cast<double>(value);
  return sum;
}

void finite_difference(nn::Tensor& value, const std::vector<float>& analytic,
                       const std::function<double()>& loss, const char* name) {
  const std::vector<float> original = value.to_host();
  CHECK(original.size() == analytic.size());
  std::vector<float> current = original;
  for (size_t i = 0; i < current.size(); ++i) {
    current[i] = original[i] + static_cast<float>(kEps);
    value.copy_from_host(current);
    const double loss_plus = loss();
    current[i] = original[i] - static_cast<float>(kEps);
    value.copy_from_host(current);
    const double loss_minus = loss();
    current[i] = original[i];
    value.copy_from_host(current);
    const double numeric = (loss_plus - loss_minus) / (2.0 * kEps);
    check_close(static_cast<double>(analytic[i]), numeric, 1e-2, 1e-4, name, i);
  }
}

void check_layer(nn::Layer& layer, nn::Tensor& input) {
  layer.zero_grad();
  const nn::Tensor& output = layer.forward(input);
  const std::vector<float> output_host = output.to_host();
  nn::Tensor grad_output(input.context(), output.shape());
  grad_output.copy_from_host(output_host);
  const std::vector<float> grad_output_before = grad_output.to_host();
  const nn::Tensor& grad_input = layer.backward(grad_output);
  const std::vector<float> analytic_input = grad_input.to_host();
  std::vector<std::pair<nn::Tensor*, std::vector<float>>> analytic_params;
  std::vector<std::string> names;
  for (const nn::NamedParam& param : layer.parameters()) {
    names.push_back(param.name);
    analytic_params.push_back({param.value, param.grad->to_host()});
  }
  const std::vector<float> grad_output_after = grad_output.to_host();
  CHECK(std::memcmp(grad_output_before.data(), grad_output_after.data(),
                    grad_output_before.size() * sizeof(float)) == 0);

  const auto loss = [&]() { return squared_loss(layer.forward(input)); };
  finite_difference(input, analytic_input, loss, "input");
  for (size_t i = 0; i < analytic_params.size(); ++i) {
    finite_difference(*analytic_params[i].first, analytic_params[i].second, loss, names[i].c_str());
  }
}

std::vector<float> uniform(int64_t n, std::mt19937& rng, float low, float high) {
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(static_cast<size_t>(n));
  for (float& value : values) value = dist(rng);
  return values;
}

void condition_bn(nn::Layer& layer) {
  // Keep every ReLU input away from zero so a 1e-3 probe stays on one side of the kink,
  // and keep the outputs moderate so float32 central differences stay inside 1e-2.
  for (nn::NamedParam param : layer.parameters()) {
    const std::string& name = param.name;
    if (name.find("bn") == std::string::npos) continue;
    if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0) param.value->fill(0.05f);
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0) param.value->fill(1.f);
  }
}

}  // namespace

int test_gradients() {
  nn::Context ctx;
  std::mt19937 rng(1);

  std::printf("grad linear\n");
  {
    nn::Linear layer(ctx, 5, 3, rng);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear no bias\n");
  {
    nn::Linear layer(ctx, 5, 3, false, rng);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear rank3\n");
  {
    nn::Linear layer(ctx, 4, 3, true, rng);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad linear rank3 no bias\n");
  {
    nn::Linear layer(ctx, 4, 3, false, rng);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad conv\n");
  {
    nn::Conv2d layer(ctx, 2, 2, 3, 1, 1, true, rng);
    nn::Tensor input(ctx, {2, 2, 5, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad batchnorm\n");
  {
    nn::BatchNorm2d layer(ctx, 2);
    nn::Tensor input(ctx, {4, 2, 3, 3});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad relu\n");
  {
    nn::ReLU layer(ctx);
    nn::Tensor input(ctx, {2, 2, 2, 2});
    std::vector<float> values(static_cast<size_t>(input.numel()));
    std::uniform_real_distribution<float> dist(0.2f, 1.f);
    for (size_t i = 0; i < values.size(); ++i) {
      const float magnitude = dist(rng);
      values[i] = (i % 2 == 0) ? magnitude : -magnitude;
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad relu rank3\n");
  {
    nn::ReLU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    std::vector<float> values(static_cast<size_t>(input.numel()));
    std::uniform_real_distribution<float> dist(0.2f, 1.f);
    for (size_t i = 0; i < values.size(); ++i) {
      const float magnitude = dist(rng);
      values[i] = (i % 2 == 0) ? magnitude : -magnitude;
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad gelu\n");
  {
    nn::GELU layer(ctx);
    nn::Tensor input(ctx, {2, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad gelu rank3\n");
  {
    nn::GELU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad layernorm\n");
  {
    nn::LayerNorm layer(ctx, 5);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad layernorm rank3\n");
  {
    nn::LayerNorm layer(ctx, 4);
    std::vector<nn::NamedParam> params = layer.parameters();
    params[0].value->copy_from_host(std::vector<float>{0.5f, -1.f, 1.5f, 0.25f});
    params[1].value->copy_from_host(std::vector<float>{0.1f, -0.2f, 0.3f, -0.4f});
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout p0\n");
  {
    nn::Dropout layer(ctx, 0.f);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout p0 rank3\n");
  {
    nn::Dropout layer(ctx, 0.f);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout eval\n");
  {
    nn::Dropout layer(ctx, 0.5f);
    layer.set_training(false);
    nn::Tensor input(ctx, {4, 5});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad dropout eval rank3\n");
  {
    nn::Dropout layer(ctx, 0.4f);
    layer.set_training(false);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 6});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu rank3\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad swiglu nchw\n");
  {
    nn::SwiGLU layer(ctx);
    nn::Tensor input(ctx, {2, 2, 2, 2});
    input.copy_from_host(uniform(input.numel(), rng, -0.8f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad last step\n");
  {
    nn::LastStep layer;
    nn::Tensor input(ctx, {2, 3, 2});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad rnn\n");
  {
    nn::RNN layer(ctx, 3, 2, rng);
    for (nn::NamedParam param : layer.parameters()) {
      if (param.name.find("weight") != std::string::npos) param.value->fill(0.2f);
    }
    nn::Tensor input(ctx, {2, 3, 3});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad maxpool\n");
  {
    nn::MaxPool2d layer(ctx, 2, 2);
    nn::Tensor input(ctx, {2, 1, 4, 4});
    std::vector<float> values(static_cast<size_t>(input.numel()), 0.f);
    const float window[4] = {0.05f, 0.15f, 0.25f, 0.80f};
    for (int n = 0; n < 2; ++n) {
      for (int by = 0; by < 2; ++by) {
        for (int bx = 0; bx < 2; ++bx) {
          for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
              const int y = by * 2 + ky;
              const int x = bx * 2 + kx;
              const size_t index = static_cast<size_t>(((n * 4) + y) * 4 + x);
              values[index] = window[ky * 2 + kx] + 0.01f * static_cast<float>(by * 2 + bx);
            }
          }
        }
      }
    }
    input.copy_from_host(values);
    check_layer(layer, input);
  }

  std::printf("grad residual identity\n");
  {
    nn::ResidualBlock layer(ctx, 2, 2, 1, rng);
    condition_bn(layer);
    nn::Tensor input(ctx, {2, 2, 6, 6});
    input.copy_from_host(uniform(input.numel(), rng, 0.2f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad residual projection\n");
  {
    nn::ResidualBlock layer(ctx, 2, 4, 2, rng);
    condition_bn(layer);
    nn::Tensor input(ctx, {2, 2, 6, 6});
    input.copy_from_host(uniform(input.numel(), rng, 0.2f, 0.8f));
    check_layer(layer, input);
  }

  std::printf("grad mha causal\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, true, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha causal no bias\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, true, false, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad mha no causal\n");
  {
    nn::MultiHeadAttention layer(ctx, 4, 2, false, false, false, rng);
    nn::Tensor input(ctx, {1, 3, 4});
    input.copy_from_host(uniform(input.numel(), rng, -0.5f, 0.5f));
    check_layer(layer, input);
  }

  std::printf("grad softmax\n");
  {
    constexpr int n = 4;
    constexpr int classes = 5;
    nn::Tensor logits(ctx, {n, classes});
    const std::vector<float> host_logits = uniform(logits.numel(), rng, -1.f, 1.f);
    logits.copy_from_host(host_logits);
    const std::vector<int64_t> host_labels = {0, 2, 4, 1};
    nn::DeviceIndices labels(ctx, n);
    labels.copy_from_host(host_labels.data(), n);
    nn::LossResult result = nn::softmax_cross_entropy(ctx, logits, labels);

    double host_loss = 0.0;
    std::vector<double> host_grad(static_cast<size_t>(n * classes), 0.0);
    for (int row = 0; row < n; ++row) {
      double max_logit = -1e300;
      for (int c = 0; c < classes; ++c) {
        max_logit = std::max(max_logit, static_cast<double>(host_logits[row * classes + c]));
      }
      double sum = 0.0;
      for (int c = 0; c < classes; ++c) {
        sum += std::exp(static_cast<double>(host_logits[row * classes + c]) - max_logit);
      }
      const double log_z = max_logit + std::log(sum);
      host_loss += -(static_cast<double>(host_logits[row * classes + host_labels[row]]) - log_z);
      for (int c = 0; c < classes; ++c) {
        const double softmax =
            std::exp(static_cast<double>(host_logits[row * classes + c]) - max_logit) / sum;
        const double indicator = (c == host_labels[row]) ? 1.0 : 0.0;
        host_grad[row * classes + c] = (softmax - indicator) / static_cast<double>(n);
      }
    }
    host_loss /= static_cast<double>(n);
    check_close(host_loss, static_cast<double>(result.loss), 1e-5, -1.0, "loss", 0);
    const std::vector<float> device_grad = result.grad_logits.to_host();
    CHECK(device_grad.size() == host_grad.size());
    for (size_t i = 0; i < host_grad.size(); ++i) {
      check_close(host_grad[i], static_cast<double>(device_grad[i]), 1e-4, -1.0, "grad_logits", i);
    }
  }

  std::printf("grads ok\n");
  return 0;
}
