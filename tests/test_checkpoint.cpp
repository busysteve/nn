#include "check.hpp"

#include "nn/builders.hpp"
#include "nn/conv_resnet.hpp"
#include "nn/error.hpp"
#include "nn/loss.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace {

std::vector<float> uniform(int64_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> values(static_cast<size_t>(n));
  for (float& value : values) value = dist(rng);
  return values;
}

bool same_bits(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  return std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

void expect_bits(const std::vector<float>& left, const std::vector<float>& right, const char* name) {
  if (!same_bits(left, right)) {
    std::fprintf(stderr, "bitwise mismatch: %s (%zu vs %zu)\n", name, left.size(), right.size());
    CHECK(false);
  }
}

void expect_throw(const std::function<void()>& call, const std::string& message) {
  bool threw = false;
  try {
    call();
  } catch (const nn::Error& error) {
    threw = true;
    if (error.what() != message) {
      std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", message.c_str(), error.what());
      CHECK(false);
    }
  }
  CHECK(threw);
}

struct Batch {
  std::vector<int> shape;
  std::vector<float> input;
  std::vector<int64_t> labels;
};

void train_step(nn::Network& net, nn::Sgd& optimizer, nn::Context& ctx, const Batch& batch) {
  net.train();
  net.zero_grad();
  nn::Tensor input(ctx, batch.shape);
  input.copy_from_host(batch.input);
  nn::DeviceIndices labels(ctx, static_cast<int64_t>(batch.labels.size()));
  labels.copy_from_host(batch.labels.data(), labels.size());
  const nn::Tensor& logits = net.forward(input);
  nn::LossResult loss = nn::softmax_cross_entropy(ctx, logits, labels);
  net.backward(loss.grad_logits);
  optimizer.step();
}

std::vector<float> eval_logits(nn::Network& net, nn::Context& ctx, const Batch& batch) {
  net.eval();
  nn::Tensor input(ctx, batch.shape);
  input.copy_from_host(batch.input);
  return net.forward(input).to_host();
}

void compare_state(const nn::Network& left, const nn::Network& right, const nn::Sgd& left_opt,
                   const nn::Sgd& right_opt, bool running_stats) {
  const std::vector<nn::NamedParam> a = left.parameters();
  const std::vector<nn::NamedParam> b = right.parameters();
  CHECK(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].name == b[i].name);
    expect_bits(a[i].value->to_host(), b[i].value->to_host(), a[i].name.c_str());
  }
  if (running_stats) {
    const std::vector<nn::NamedBuffer> ab = left.buffers();
    const std::vector<nn::NamedBuffer> bb = right.buffers();
    CHECK(ab.size() == bb.size());
    for (size_t i = 0; i < ab.size(); ++i) {
      CHECK(ab[i].name == bb[i].name);
      CHECK(*ab[i].batches == *bb[i].batches);
      expect_bits(ab[i].value->to_host(), bb[i].value->to_host(), ab[i].name.c_str());
    }
  }
  CHECK(left_opt.step_count() == right_opt.step_count());
  CHECK(left_opt.learning_rate() == right_opt.learning_rate());
  CHECK(left_opt.momentum() == right_opt.momentum());
  CHECK(left_opt.velocity().size() == right_opt.velocity().size());
  for (size_t i = 0; i < left_opt.velocity().size(); ++i) {
    expect_bits(left_opt.velocity()[i].to_host(), right_opt.velocity()[i].to_host(), "velocity");
  }
}

void round_trip(nn::Network net, nn::Context& ctx_a, const std::filesystem::path& path, const Batch& train,
                const Batch& probe, bool running_stats) {
  net.train();
  nn::Sgd optimizer(net.parameters(), 0.1f, 0.9f);
  train_step(net, optimizer, ctx_a, train);
  CHECK(optimizer.step_count() == 1);
  const std::vector<float> logits = eval_logits(net, ctx_a, probe);
  net.save(path, &optimizer);

  nn::Context ctx_b;
  nn::Network::Loaded loaded = nn::Network::load(ctx_b, path, nn::LoadMode::Resume);
  CHECK(loaded.optimizer.has_value());
  CHECK(loaded.step == 1);
  CHECK(!loaded.network.training());
  compare_state(net, loaded.network, optimizer, *loaded.optimizer, running_stats);
  expect_bits(logits, eval_logits(loaded.network, ctx_b, probe), "logits");

  train_step(net, optimizer, ctx_a, train);
  train_step(loaded.network, *loaded.optimizer, ctx_b, train);
  compare_state(net, loaded.network, optimizer, *loaded.optimizer, running_stats);
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

void write_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  bytes[offset] = static_cast<uint8_t>(value);
  bytes[offset + 1] = static_cast<uint8_t>(value >> 8);
  bytes[offset + 2] = static_cast<uint8_t>(value >> 16);
  bytes[offset + 3] = static_cast<uint8_t>(value >> 24);
}

void write_u64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  write_u32(bytes, offset, static_cast<uint32_t>(value));
  write_u32(bytes, offset + 4, static_cast<uint32_t>(value >> 32));
}

size_t align8(size_t offset) { return (offset + 7u) & ~size_t{7}; }

size_t skip_string(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t length = read_u32(bytes, offset);
  return align8(offset + 4 + length);
}

size_t skip_shape(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t rank = read_u32(bytes, offset);
  return align8(offset + 4 + 4 * rank);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(static_cast<bool>(in));
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  CHECK(static_cast<bool>(in));
  return bytes;
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  CHECK(static_cast<bool>(out));
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  CHECK(static_cast<bool>(out));
}

size_t optimizer_offset(const std::vector<uint8_t>& bytes) {
  size_t pos = 32;
  const uint32_t hidden = read_u32(bytes, pos + 12);
  pos = align8(pos + 16 + 4 * hidden);
  const uint32_t param_count = read_u32(bytes, pos);
  pos += 8;
  for (uint32_t i = 0; i < param_count; ++i) {
    pos = skip_string(bytes, pos);
    pos = skip_shape(bytes, pos);
    const uint32_t numel = read_u32(bytes, pos + 4);
    pos = align8(pos + 8 + 4 * static_cast<size_t>(numel));
  }
  const uint32_t bn_count = read_u32(bytes, pos);
  CHECK(bn_count == 0);
  pos += 8;
  return pos;
}

}  // namespace

int test_checkpoint() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "nn-ckpt";
  std::filesystem::create_directories(dir);
  nn::Context ctx;
  std::mt19937 rng(1);

  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp.bin", train, probe, false);
  }
  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    config.activation = nn::Activation::Gelu;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp-gelu.bin", train, probe, false);
    nn::Network::Loaded loaded =
        nn::Network::load(ctx, dir / "mlp-gelu.bin", nn::LoadMode::Inference);
    CHECK(std::get<nn::MlpConfig>(loaded.network.config()).activation == nn::Activation::Gelu);
  }
  {
    nn::MlpConfig config;
    config.input_size = 8;
    config.hidden = {6};
    config.output_size = 4;
    config.activation = nn::Activation::SwiGLU;
    Batch train{{2, 8}, uniform(16, rng), {0, 1}};
    Batch probe{{2, 8}, uniform(16, rng), {1, 0}};
    round_trip(nn::build_mlp(ctx, config, 1), ctx, dir / "mlp-swiglu.bin", train, probe, false);
    nn::Network::Loaded loaded =
        nn::Network::load(ctx, dir / "mlp-swiglu.bin", nn::LoadMode::Inference);
    CHECK(std::get<nn::MlpConfig>(loaded.network.config()).activation == nn::Activation::SwiGLU);
    CHECK(loaded.network.parameters()[0].value->shape() == std::vector<int>({12, 8}));
  }
  {
    nn::ConvConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.convs = {nn::ConvSpec{3, 3, 1, 1}};
    config.pool = nn::PoolSpec{2, 2};
    config.fc = {4, 4};
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {1, 2}};
    round_trip(nn::build_conv(ctx, config, 1), ctx, dir / "conv.bin", train, probe, false);
  }
  {
    nn::ResNetConfig config;
    config.in_channels = 1;
    config.in_height = 8;
    config.in_width = 8;
    config.stem_channels = 4;
    config.blocks_per_stage = {1, 1};
    config.stage_widths = {4, 8};
    config.output_size = 4;
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {2, 3}};
    round_trip(nn::build_resnet(ctx, config, 1), ctx, dir / "resnet.bin", train, probe, true);
  }
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
    Batch train{{2, 1, 8, 8}, uniform(2 * 64, rng), {0, 1}};
    Batch probe{{2, 1, 8, 8}, uniform(2 * 64, rng), {2, 3}};
    const std::filesystem::path path = dir / "convresnet.bin";
    round_trip(nn::build_conv_resnet(ctx, config, 1), ctx, path, train, probe, true);

    nn::ConvResNet::Loaded loaded = nn::ConvResNet::load(ctx, path, nn::LoadMode::Resume);
    CHECK(loaded.model.family() == nn::Family::ConvResNet);
    CHECK(loaded.optimizer.has_value());
    CHECK(loaded.step == 1);
    CHECK(loaded.model.config().fc == config.fc);
    CHECK(loaded.model.config().convs.size() == 1);
    CHECK(loaded.model.config().blocks_per_stage.size() == 2);
  }

  const std::filesystem::path mlp = dir / "mlp.bin";
  {
    const std::vector<uint8_t> bytes = read_bytes(mlp);
    write_bytes(dir / "trunc.bin", std::vector<uint8_t>(bytes.begin(), bytes.begin() + 16));
    expect_throw([&] { nn::Network::load(ctx, dir / "trunc.bin", nn::LoadMode::Inference); },
                 "truncated checkpoint");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    write_u32(bytes, 4, 99);
    write_bytes(dir / "version.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "version.bin", nn::LoadMode::Inference); },
                 "unsupported checkpoint version 99");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    const std::string name = "fc.0.weight";
    const auto found = std::search(bytes.begin(), bytes.end(), name.begin(), name.end());
    CHECK(found != bytes.end());
    const size_t name_at = static_cast<size_t>(found - bytes.begin());
    CHECK(read_u32(bytes, name_at - 4) == name.size());
    size_t pos = align8(name_at + name.size());
    CHECK(read_u32(bytes, pos) == 2);
    const int32_t dim = static_cast<int32_t>(read_u32(bytes, pos + 4));
    write_u32(bytes, pos + 4, static_cast<uint32_t>(dim + 1));
    write_bytes(dir / "shape.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "shape.bin", nn::LoadMode::Inference); },
                 "checkpoint tensor mismatch: fc.0.weight");
  }
  {
    std::vector<uint8_t> bytes = read_bytes(mlp);
    bytes.resize(optimizer_offset(bytes));
    write_u64(bytes, 16, 0);
    write_u32(bytes, 24, 0);
    write_bytes(dir / "noopt.bin", bytes);
    expect_throw([&] { nn::Network::load(ctx, dir / "noopt.bin", nn::LoadMode::Resume); },
                 "optimizer block missing");
  }
  {
    nn::Network::Loaded loaded = nn::Network::load(ctx, mlp, nn::LoadMode::Inference);
    CHECK(!loaded.optimizer.has_value());
    CHECK(loaded.step == 1);
    expect_throw([&] { nn::ConvResNet::load(ctx, mlp, nn::LoadMode::Inference); },
                 "bad checkpoint family");
    CHECK(!loaded.network.training());
  }

  std::printf("ckpt ok\n");
  return 0;
}
