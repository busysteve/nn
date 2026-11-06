#include "nn/builders.hpp"
#include "nn/error.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nn {
namespace {

constexpr uint32_t kMagic = 0x314E4E43u;
constexpr uint32_t kVersion = 1u;
constexpr uint32_t kFloat32 = 1u;

class ByteWriter {
 public:
  void u32(uint32_t value) {
    bytes_.push_back(static_cast<uint8_t>(value));
    bytes_.push_back(static_cast<uint8_t>(value >> 8));
    bytes_.push_back(static_cast<uint8_t>(value >> 16));
    bytes_.push_back(static_cast<uint8_t>(value >> 24));
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void i32(int32_t value) { u32(static_cast<uint32_t>(value)); }
  void f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }
  void pad8() {
    while (bytes_.size() % 8 != 0) bytes_.push_back(0);
  }
  void str(const std::string& value) {
    if (value.size() > 256) throw Error("checkpoint tensor mismatch: name");
    u32(static_cast<uint32_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
    pad8();
  }
  void shape(const std::vector<int>& dims) {
    u32(static_cast<uint32_t>(dims.size()));
    for (int dim : dims) i32(dim);
    pad8();
  }
  void f32s(const std::vector<float>& values) {
    for (float value : values) f32(value);
  }
  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

class ByteReader {
 public:
  explicit ByteReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

  uint32_t u32() {
    need(4);
    uint32_t value = static_cast<uint32_t>(bytes_[pos_]) |
                     (static_cast<uint32_t>(bytes_[pos_ + 1]) << 8) |
                     (static_cast<uint32_t>(bytes_[pos_ + 2]) << 16) |
                     (static_cast<uint32_t>(bytes_[pos_ + 3]) << 24);
    pos_ += 4;
    return value;
  }
  uint64_t u64() {
    const uint64_t low = u32();
    const uint64_t high = u32();
    return low | (high << 32);
  }
  int32_t i32() { return static_cast<int32_t>(u32()); }
  float f32() {
    const uint32_t bits = u32();
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  void pad8() {
    while (pos_ % 8 != 0) {
      need(1);
      ++pos_;
    }
  }
  std::string str() {
    const uint32_t len = u32();
    if (len > 256) throw Error("checkpoint tensor mismatch: name");
    need(len);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + pos_), len);
    pos_ += len;
    pad8();
    return value;
  }
  std::vector<int> shape() {
    const uint32_t rank = u32();
    if (rank != 1 && rank != 2 && rank != 4) throw Error("checkpoint tensor mismatch: rank");
    if (rank > (bytes_.size() - pos_) / 4) throw Error("truncated checkpoint");
    std::vector<int> dims(rank);
    for (uint32_t i = 0; i < rank; ++i) dims[i] = i32();
    pad8();
    return dims;
  }
  std::vector<float> f32s(uint32_t count) {
    if (static_cast<uint64_t>(count) > (bytes_.size() - pos_) / 4) throw Error("truncated checkpoint");
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) values[i] = f32();
    return values;
  }

 private:
  void need(size_t n) const {
    if (pos_ + n > bytes_.size()) throw Error("truncated checkpoint");
  }
  std::vector<uint8_t> bytes_;
  size_t pos_ = 0;
};

struct ParsedParam {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
};

struct ParsedBn {
  std::string prefix;
  uint64_t batches = 0;
  std::vector<float> mean;
  std::vector<float> var;
};

struct ParsedOpt {
  float learning_rate = 0.f;
  float momentum = 0.f;
  std::vector<std::vector<float>> velocity;
};

struct Header {
  uint32_t family = 0;
  uint32_t training = 0;
  uint64_t step = 0;
  uint32_t optimizer_present = 0;
};

int64_t product(const std::vector<int>& shape) {
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) return -1;
    n *= dim;
  }
  return n;
}

std::string strip_suffix(const std::string& name, const std::string& suffix) {
  if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return {};
  }
  return name.substr(0, name.size() - suffix.size());
}

void write_all(int fd, const uint8_t* data, size_t n) {
  while (n > 0) {
    const ssize_t wrote = ::write(fd, data, n);
    if (wrote < 0) {
      if (errno == EINTR) continue;
      throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    }
    data += wrote;
    n -= static_cast<size_t>(wrote);
  }
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("truncated checkpoint");
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 32) throw Error("truncated checkpoint");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) throw Error("truncated checkpoint");
  return bytes;
}

Header read_header(ByteReader& reader) {
  const uint32_t magic = reader.u32();
  if (magic != kMagic) throw Error("bad checkpoint magic");
  const uint32_t version = reader.u32();
  if (version != kVersion) {
    throw Error("unsupported checkpoint version " + std::to_string(version));
  }
  Header header;
  header.family = reader.u32();
  header.training = reader.u32();
  header.step = reader.u64();
  header.optimizer_present = reader.u32();
  const uint32_t reserved = reader.u32();
  if (reserved != 0 || header.training > 1 || header.optimizer_present > 1) {
    throw Error("bad checkpoint header");
  }
  if (header.family < 1 || header.family > 4) throw Error("bad checkpoint family");
  return header;
}

std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig> read_config(ByteReader& reader,
                                                                                 uint32_t family) {
  if (family == 1) {
    MlpConfig config;
    config.input_size = reader.i32();
    config.output_size = reader.i32();
    config.activation = static_cast<Activation>(reader.u32());
    const uint32_t hidden_count = reader.u32();
    if (hidden_count > 1000000u) throw Error("truncated checkpoint");
    config.hidden.resize(hidden_count);
    for (uint32_t i = 0; i < hidden_count; ++i) config.hidden[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 2) {
    ConvConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.pool.kernel = reader.i32();
    config.pool.stride = reader.i32();
    const uint32_t conv_count = reader.u32();
    if (conv_count > 1000000u) throw Error("truncated checkpoint");
    config.convs.resize(conv_count);
    for (uint32_t i = 0; i < conv_count; ++i) {
      config.convs[i].out_channels = reader.i32();
      config.convs[i].kernel = reader.i32();
      config.convs[i].stride = reader.i32();
      config.convs[i].pad = reader.i32();
    }
    const uint32_t fc_count = reader.u32();
    if (fc_count > 1000000u) throw Error("truncated checkpoint");
    config.fc.resize(fc_count);
    for (uint32_t i = 0; i < fc_count; ++i) config.fc[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 3) {
    ResNetConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.stem_channels = reader.i32();
    config.output_size = reader.i32();
    const uint32_t stage_count = reader.u32();
    if (stage_count > 1000000u) throw Error("truncated checkpoint");
    config.blocks_per_stage.resize(stage_count);
    config.stage_widths.resize(stage_count);
    for (uint32_t i = 0; i < stage_count; ++i) {
      config.blocks_per_stage[i] = reader.i32();
      config.stage_widths[i] = reader.i32();
    }
    reader.pad8();
    validate(config);
    return config;
  }
  if (family == 4) {
    ConvResNetConfig config;
    config.in_channels = reader.i32();
    config.in_height = reader.i32();
    config.in_width = reader.i32();
    config.stem_channels = reader.i32();
    config.pool.kernel = reader.i32();
    config.pool.stride = reader.i32();
    const uint32_t conv_count = reader.u32();
    if (conv_count > 1000000u) throw Error("truncated checkpoint");
    config.convs.resize(conv_count);
    for (uint32_t i = 0; i < conv_count; ++i) {
      config.convs[i].out_channels = reader.i32();
      config.convs[i].kernel = reader.i32();
      config.convs[i].stride = reader.i32();
      config.convs[i].pad = reader.i32();
    }
    const uint32_t stage_count = reader.u32();
    if (stage_count > 1000000u) throw Error("truncated checkpoint");
    config.blocks_per_stage.resize(stage_count);
    config.stage_widths.resize(stage_count);
    for (uint32_t i = 0; i < stage_count; ++i) {
      config.blocks_per_stage[i] = reader.i32();
      config.stage_widths[i] = reader.i32();
    }
    const uint32_t fc_count = reader.u32();
    if (fc_count > 1000000u) throw Error("truncated checkpoint");
    config.fc.resize(fc_count);
    for (uint32_t i = 0; i < fc_count; ++i) config.fc[i] = reader.i32();
    reader.pad8();
    validate(config);
    return config;
  }
  throw Error("bad checkpoint family");
}

std::vector<ParsedParam> read_params(ByteReader& reader, uint32_t* count_out) {
  const uint32_t count = reader.u32();
  reader.u32();
  if (count > 1000000u) throw Error("truncated checkpoint");
  std::vector<ParsedParam> params(count);
  for (uint32_t i = 0; i < count; ++i) {
    params[i].name = reader.str();
    params[i].shape = reader.shape();
    const uint32_t dtype = reader.u32();
    const uint32_t numel = reader.u32();
    if (dtype != kFloat32 || product(params[i].shape) != static_cast<int64_t>(numel)) {
      throw Error("checkpoint tensor mismatch: " + params[i].name);
    }
    params[i].data = reader.f32s(numel);
    reader.pad8();
  }
  *count_out = count;
  return params;
}

std::vector<ParsedBn> read_bn(ByteReader& reader) {
  const uint32_t count = reader.u32();
  reader.u32();
  if (count > 1000000u) throw Error("truncated checkpoint");
  std::vector<ParsedBn> layers(count);
  for (uint32_t i = 0; i < count; ++i) {
    layers[i].prefix = reader.str();
    layers[i].batches = reader.u64();
    const uint32_t channels = reader.u32();
    reader.u32();
    if (channels == 0 || channels > 1000000u) throw Error("checkpoint tensor mismatch: " + layers[i].prefix);
    layers[i].mean = reader.f32s(channels);
    layers[i].var = reader.f32s(channels);
    reader.pad8();
  }
  return layers;
}

ParsedOpt read_opt(ByteReader& reader, const std::vector<ParsedParam>& params) {
  ParsedOpt opt;
  opt.learning_rate = reader.f32();
  opt.momentum = reader.f32();
  const uint32_t count = reader.u32();
  reader.u32();
  if (count != params.size()) throw Error("optimizer state mismatch");
  opt.velocity.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t numel = reader.u32();
    reader.u32();
    if (numel != params[i].data.size()) throw Error("optimizer state mismatch");
    opt.velocity[i] = reader.f32s(numel);
    reader.pad8();
  }
  return opt;
}

void require_match(const ParsedParam& parsed, const NamedParam& param) {
  if (parsed.name != param.name || parsed.shape != param.value->shape() ||
      static_cast<int64_t>(parsed.data.size()) != param.value->numel()) {
    throw Error("checkpoint tensor mismatch: " + parsed.name);
  }
}

}  // namespace

void Network::save(const std::filesystem::path& path, const Sgd* optimizer) const {
  if (optimizer) {
    const std::vector<NamedParam> mine = parameters();
    const std::vector<NamedParam>& theirs = optimizer->params();
    if (mine.size() != theirs.size()) throw Error("optimizer does not belong to this network");
    for (size_t i = 0; i < mine.size(); ++i) {
      if (mine[i].value != theirs[i].value) throw Error("optimizer does not belong to this network");
    }
  }
  ctx_->synchronize();
  const std::vector<NamedParam> params = parameters();
  const std::vector<NamedBuffer> buffers = this->buffers();
  if (buffers.size() % 2 != 0) throw Error("checkpoint tensor mismatch: batch-norm buffers");

  ByteWriter writer;
  writer.u32(kMagic);
  writer.u32(kVersion);
  writer.u32(static_cast<uint32_t>(family_));
  writer.u32(training_ ? 1u : 0u);
  writer.u64(optimizer ? optimizer->step_count() : 0);
  writer.u32(optimizer ? 1u : 0u);
  writer.u32(0);

  std::visit(
      [&](const auto& config) {
        using T = std::decay_t<decltype(config)>;
        if constexpr (std::is_same_v<T, MlpConfig>) {
          writer.i32(config.input_size);
          writer.i32(config.output_size);
          writer.u32(static_cast<uint32_t>(config.activation));
          writer.u32(static_cast<uint32_t>(config.hidden.size()));
          for (int width : config.hidden) writer.i32(width);
        } else if constexpr (std::is_same_v<T, ConvConfig>) {
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.pool.kernel);
          writer.i32(config.pool.stride);
          writer.u32(static_cast<uint32_t>(config.convs.size()));
          for (const ConvSpec& spec : config.convs) {
            writer.i32(spec.out_channels);
            writer.i32(spec.kernel);
            writer.i32(spec.stride);
            writer.i32(spec.pad);
          }
          writer.u32(static_cast<uint32_t>(config.fc.size()));
          for (int width : config.fc) writer.i32(width);
        } else if constexpr (std::is_same_v<T, ResNetConfig>) {
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.stem_channels);
          writer.i32(config.output_size);
          writer.u32(static_cast<uint32_t>(config.blocks_per_stage.size()));
          for (size_t i = 0; i < config.blocks_per_stage.size(); ++i) {
            writer.i32(config.blocks_per_stage[i]);
            writer.i32(config.stage_widths[i]);
          }
        } else {
          static_assert(std::is_same_v<T, ConvResNetConfig>);
          writer.i32(config.in_channels);
          writer.i32(config.in_height);
          writer.i32(config.in_width);
          writer.i32(config.stem_channels);
          writer.i32(config.pool.kernel);
          writer.i32(config.pool.stride);
          writer.u32(static_cast<uint32_t>(config.convs.size()));
          for (const ConvSpec& spec : config.convs) {
            writer.i32(spec.out_channels);
            writer.i32(spec.kernel);
            writer.i32(spec.stride);
            writer.i32(spec.pad);
          }
          writer.u32(static_cast<uint32_t>(config.blocks_per_stage.size()));
          for (size_t i = 0; i < config.blocks_per_stage.size(); ++i) {
            writer.i32(config.blocks_per_stage[i]);
            writer.i32(config.stage_widths[i]);
          }
          writer.u32(static_cast<uint32_t>(config.fc.size()));
          for (int width : config.fc) writer.i32(width);
        }
      },
      config_);
  writer.pad8();

  writer.u32(static_cast<uint32_t>(params.size()));
  writer.u32(0);
  for (const NamedParam& param : params) {
    const std::vector<float> values = param.value->to_host();
    writer.str(param.name);
    writer.shape(param.value->shape());
    writer.u32(kFloat32);
    writer.u32(static_cast<uint32_t>(values.size()));
    writer.f32s(values);
    writer.pad8();
  }

  writer.u32(static_cast<uint32_t>(buffers.size() / 2));
  writer.u32(0);
  for (size_t i = 0; i < buffers.size(); i += 2) {
    const std::string mean_prefix = strip_suffix(buffers[i].name, ".running_mean");
    const std::string var_prefix = strip_suffix(buffers[i + 1].name, ".running_var");
    if (mean_prefix.empty() || mean_prefix != var_prefix) {
      throw Error("checkpoint tensor mismatch: " + buffers[i].name);
    }
    const std::vector<float> mean = buffers[i].value->to_host();
    const std::vector<float> var = buffers[i + 1].value->to_host();
    writer.str(mean_prefix);
    writer.u64(*buffers[i].batches);
    writer.u32(static_cast<uint32_t>(mean.size()));
    writer.u32(0);
    writer.f32s(mean);
    writer.f32s(var);
    writer.pad8();
  }

  if (optimizer) {
    writer.f32(optimizer->learning_rate());
    writer.f32(optimizer->momentum());
    writer.u32(static_cast<uint32_t>(optimizer->velocity().size()));
    writer.u32(0);
    for (const Tensor& velocity : optimizer->velocity()) {
      const std::vector<float> values = velocity.to_host();
      writer.u32(static_cast<uint32_t>(values.size()));
      writer.u32(0);
      writer.f32s(values);
      writer.pad8();
    }
  }

  const std::filesystem::path tmp = path.string() + ".tmp";
  int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
  try {
    write_all(fd, writer.bytes().data(), writer.bytes().size());
    if (::fsync(fd) != 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    if (::close(fd) != 0) throw Error(std::string("checkpoint write: ") + std::strerror(errno));
    fd = -1;
    std::filesystem::rename(tmp, path);
  } catch (...) {
    if (fd >= 0) ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
}

Network::Loaded Network::load(Context& ctx, const std::filesystem::path& path, LoadMode mode) {
  ByteReader reader(read_file(path));
  const Header header = read_header(reader);
  auto config = read_config(reader, header.family);
  uint32_t param_count = 0;
  const std::vector<ParsedParam> parsed_params = read_params(reader, &param_count);
  const std::vector<ParsedBn> parsed_bn = read_bn(reader);
  std::optional<ParsedOpt> parsed_opt;
  if (header.optimizer_present) parsed_opt = read_opt(reader, parsed_params);
  if (mode == LoadMode::Resume && !parsed_opt) throw Error("optimizer block missing");

  Network network = std::visit(
      [&](const auto& cfg) -> Network {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, MlpConfig>) return build_mlp(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, ConvConfig>) return build_conv(ctx, cfg, 0);
        else if constexpr (std::is_same_v<T, ResNetConfig>) return build_resnet(ctx, cfg, 0);
        else return build_conv_resnet(ctx, cfg, 0);
      },
      config);

  const std::vector<NamedParam> params = network.parameters();
  if (params.size() != parsed_params.size()) throw Error("checkpoint tensor mismatch: count");
  for (size_t i = 0; i < params.size(); ++i) require_match(parsed_params[i], params[i]);

  const std::vector<NamedBuffer> buffers = network.buffers();
  if (buffers.size() != parsed_bn.size() * 2) throw Error("checkpoint tensor mismatch: batch-norm buffers");
  for (size_t i = 0; i < parsed_bn.size(); ++i) {
    const std::string mean_prefix = strip_suffix(buffers[2 * i].name, ".running_mean");
    const std::string var_prefix = strip_suffix(buffers[2 * i + 1].name, ".running_var");
    if (mean_prefix != parsed_bn[i].prefix || var_prefix != parsed_bn[i].prefix ||
        buffers[2 * i].value->shape() != std::vector<int>{static_cast<int>(parsed_bn[i].mean.size())} ||
        buffers[2 * i + 1].value->shape() != std::vector<int>{static_cast<int>(parsed_bn[i].var.size())}) {
      throw Error("checkpoint tensor mismatch: " + parsed_bn[i].prefix);
    }
  }

  for (size_t i = 0; i < params.size(); ++i) params[i].value->copy_from_host(parsed_params[i].data);
  for (size_t i = 0; i < parsed_bn.size(); ++i) {
    buffers[2 * i].value->copy_from_host(parsed_bn[i].mean);
    buffers[2 * i + 1].value->copy_from_host(parsed_bn[i].var);
    *buffers[2 * i].batches = parsed_bn[i].batches;
  }
  if (header.training) network.train();
  else network.eval();

  Loaded loaded{std::move(network), std::nullopt, header.step};
  if (mode == LoadMode::Resume) {
    Sgd optimizer(loaded.network.parameters(), parsed_opt->learning_rate, parsed_opt->momentum);
    for (size_t i = 0; i < parsed_opt->velocity.size(); ++i) {
      optimizer.velocity()[i].copy_from_host(parsed_opt->velocity[i]);
    }
    optimizer.set_step_count(header.step);
    loaded.optimizer = std::move(optimizer);
  }
  ctx.synchronize();
  return loaded;
}

}  // namespace nn
