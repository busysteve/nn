# Usage

This is the programmer's guide to the public types in namespace `nn`. Every type lives in a header under `include/nn/`. Those headers are plain C++17. Build steps, numeric details, the checkpoint byte layout, and the MNIST suite are in [README.md](README.md).

Link the CMake target `nn`. It carries `CUDA::cudart` and `CUDA::cublas`. A program that links `libnn.a` by hand also links the cuDNN library the archive was built against.

Create one `Context` on the CUDA device you want, then create tensors, layers, and the optimizer from that context. Destroy those objects before the context. The context uses the device that is current when it is constructed, and it owns one stream. `synchronize()` waits for that stream.

Storage is dense float32. A rank-2 tensor is `(N, F)`, row-major. A rank-4 tensor is `(N, C, H, W)`, NCHW. `forward` and `backward` return a `const Tensor&` into storage owned by that layer. The reference stays valid until the next `forward` or `backward` on the same layer, or until the layer is destroyed. Keep values with `to_host()` or `copy_from`.

`NamedParam` and `NamedBuffer` hold raw pointers into those tensors. `parameters()` and `buffers()` return a fresh vector of pointers. The storage lives as long as the owning layer does. `Sgd` copies the pointers and owns one velocity tensor per parameter.

Library failures throw `nn::Error`. Config checks throw `std::invalid_argument` whose message is the field name.

## Train one batch

```cpp
#include "nn/builders.hpp"
#include "nn/loss.hpp"

nn::Context ctx;

nn::MlpConfig config;
config.input_size = 784;
config.hidden = {256, 128};
config.output_size = 10;

nn::Network net = nn::build_mlp(ctx, config, /*seed=*/1);
nn::Sgd optimizer(net.parameters(), /*learning_rate=*/0.1f, /*momentum=*/0.9f);

nn::Tensor input(ctx, {128, 784});
nn::DeviceIndices labels(ctx, 128);
// Fill input with copy_from_host, and labels with copy_from_host.

net.train();
net.zero_grad();
const nn::Tensor& logits = net.forward(input);
nn::LossResult loss = nn::softmax_cross_entropy(ctx, logits, labels);
net.backward(loss.grad_logits);
optimizer.step();

net.eval();
net.save("mlp.bin", nullptr);
```

Weight and bias gradients accumulate. Call `zero_grad` once per step. A new network is in train mode. Call `eval()` before inference when the network contains batch-norm.

For an image network, `input` is rank 4: `(N, in_channels, in_height, in_width)`.

## Error

Header: `nn/error.hpp`

```cpp
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message);
};
```

Catch `nn::Error` for a failed CUDA, cuDNN, or cuBLAS call, a bad tensor or layer argument, or a checkpoint problem. The message for a runtime status is `call: STATUS_NAME`, for example `cudnnConvolutionForward: CUDNN_STATUS_BAD_PARAM`. Catch `std::invalid_argument` when `validate` or `conv_out_size` rejects a config field.

## Context

Header: `nn/context.hpp`

```cpp
class Context {
 public:
  Context();
  void synchronize() const;
};
```

`Context` is neither copyable nor movable. The constructor creates the stream, the cuDNN handle, and the cuBLAS handle on the current device. Pass the same `Context&` to every tensor, label buffer, layer, and loss that must share that stream.

`synchronize()` waits until work queued on that stream has finished. `Tensor::to_host()` and `kaiming_normal` synchronize for you. `copy_from_host` queues an async copy, so call `synchronize()` before reusing the host buffer.

## Tensor

Header: `nn/tensor.hpp`

```cpp
class Tensor {
 public:
  Tensor(Context& ctx, std::vector<int> shape);
  Context& context() const;
  int rank() const;
  const std::vector<int>& shape() const;
  int64_t numel() const;
  float* data();
  const float* data() const;

  void zero();
  void fill(float value);
  void copy_from(const Tensor& src);
  void copy_from_host(const float* src, int64_t n);
  void copy_from_host(const std::vector<float>& src);
  std::vector<float> to_host() const;
  Tensor reshape(const std::vector<int>& shape) const;
};
```

`Tensor` is movable and not copyable. The constructor allocates device memory with `cudaMalloc`. Fresh memory is uninitialized. Call `zero`, `fill`, or `copy_from_host` before a read.

The shape has rank 1, 2, 3, or 4. Every dimension is at least 1. The element count must fit in a signed 32-bit integer. A bad rank throws `tensor rank out of range`. A bad dimension throws `tensor dimension out of range`. An overflowing product throws `tensor numel out of range`.

`data()` is a device pointer. Kernels that use it run on `context()`'s stream.

`copy_from` requires the same element count and throws `tensor copy numel mismatch` otherwise. A source on another context is synchronized before the copy. `copy_from_host` throws `tensor host copy numel mismatch` when `n` or `src.size()` differs from `numel()`.

`to_host()` synchronizes, then copies the tensor to a `std::vector<float>`.

`reshape` checks that the new shape has the same element count, allocates a new tensor on the same context, and copies. It throws `tensor reshape numel mismatch` when the counts differ.

The destructor synchronizes the stream and frees the device memory.

## DeviceIndices

Header: `nn/loss.hpp`

```cpp
class DeviceIndices {
 public:
  DeviceIndices(Context& ctx, int64_t n);
  void copy_from_host(const int64_t* src, int64_t n);
  const int64_t* data() const;
  int64_t size() const;
  Context& context() const;
};
```

`DeviceIndices` is the label buffer for `softmax_cross_entropy`. It is movable and not copyable. `n` is the batch size and must be positive. `n <= 0` throws `empty labels`. Values are `int64` class indices on the device. `copy_from_host` requires `n == size()` and throws `label count mismatch` otherwise. The destructor synchronizes and frees the buffer.

## LossResult

Header: `nn/loss.hpp`

```cpp
struct LossResult {
  float loss = 0.f;
  Tensor grad_logits;
};

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels);
```

`logits` is rank 2, shape `(N, C)`, with `C > 1`. `labels.size()` equals `N`. Each label is in `[0, C)`. The returned `loss` is the mean softmax cross-entropy over the batch. `grad_logits` has the same shape as `logits` and holds `(softmax - one_hot) / N`. Pass that tensor to `backward`.

| Condition | Message |
|---|---|
| Rank other than 2 | `logits rank` |
| `N <= 0` or `C <= 1` | `logits shape` |
| Label count differs from `N` | `label count mismatch` |
| A label outside `[0, C)` | `label out of range` |

`LossResult` owns `grad_logits`. It is movable and not copyable.

## NamedParam and NamedBuffer

Header: `nn/layer.hpp`

```cpp
struct NamedParam {
  std::string name;
  Tensor* value = nullptr;
  Tensor* grad = nullptr;
};

struct NamedBuffer {
  std::string name;
  Tensor* value = nullptr;
  uint64_t* batches = nullptr;
};
```

A parameter is a trainable tensor plus its gradient. A buffer is state that is saved and restored and is left alone by `zero_grad` and by `Sgd`. `batches` points at the batch-norm counter shared by that layer's `running_mean` and `running_var`.

`Layer::parameters()` returns names local to the layer, such as `weight` and `bias`. `Network::parameters()` prefixes the layer name. A layer named `fc.0` contributes `fc.0.weight` and `fc.0.bias`. An empty layer name contributes the local name unchanged. `ResidualBlock` prefixes its children first, so a block named `stages.0.blocks.1` contributes `stages.0.blocks.1.conv1.weight`.

## Layer

Header: `nn/layer.hpp`

```cpp
class Layer {
 public:
  virtual const Tensor& forward(const Tensor& input) = 0;
  virtual const Tensor& backward(const Tensor& grad_output) = 0;
  virtual std::vector<NamedParam> parameters() const = 0;
  virtual std::vector<NamedBuffer> buffers() const;
  virtual void zero_grad() = 0;
  virtual void set_training(bool);

  const std::string& name() const;
  void set_name(std::string name);
};
```

`Layer` is the abstract base. Construct a concrete layer with a `Context`, then call `forward` and `backward` yourself or hand ownership to a `Network` with `std::unique_ptr<Layer>`. `set_name` before the network is built. `buffers()` defaults to an empty list. `set_training` defaults to a no-op. `BatchNorm2d` and `ResidualBlock` honor it.

`backward` reads the activation saved by the matching `forward`. Call `forward` on that layer before `backward`.

### Linear

Header: `nn/layers/linear.hpp`

```cpp
Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng);
```

`in_features` and `out_features` are at least 1. A smaller value throws `Linear size`. The constructor draws `weight` with `kaiming_normal` and fan-in `in_features`, and sets `bias` to zero. Both draws and the zero fill use `rng` only for the weight. The bias fill consumes no random numbers.

| Tensor | Shape | Name |
|---|---|---|
| weight | `(out_features, in_features)` | `weight` |
| bias | `(out_features)` | `bias` |

`forward` takes `(N, in_features)` and returns `(N, out_features)`:

```
y = x * W^T + b
```

A rank other than 2, or a feature count other than `in_features`, throws `Linear input shape`. `backward` takes a gradient of shape `(N, out_features)`, returns `(N, in_features)`, and adds into `grad_weight` and `grad_bias`. A mismatched gradient throws `Linear backward shape`.

### ReLU

Header: `nn/layers/relu.hpp`

```cpp
explicit ReLU(Context& ctx);
```

No parameters. `forward` and `backward` accept rank 2 or rank 4 and return the same shape. Rank 2 is presented to cuDNN as `(N, 1, 1, F)`. Any other rank throws `ReLU rank`. `backward` requires the gradient shape to match the saved forward output and throws `ReLU backward shape` otherwise. `zero_grad` is empty.

### Conv2d

Header: `nn/layers/conv2d.hpp`

```cpp
Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
       bool use_bias, std::mt19937& rng);
```

The kernel is square. `in_channels` and `out_channels` are at least 1 (`Conv2d channels`). `kernel` and `stride` are at least 1, and `pad` is at least 0. A failed size check throws `std::invalid_argument` with message `kernel`, `stride`, or `pad`.

Weight shape is `(out_channels, in_channels, kernel, kernel)`. Fan-in is `in_channels * kernel * kernel`. Bias, when `use_bias` is true, has shape `(out_channels)` and starts at zero. Parameter names are `weight`, then `bias` when bias is on.

`forward` takes `(N, in_channels, H, W)` and returns `(N, out_channels, H_out, W_out)`, where

```
H_out = (H + 2 * pad - kernel) / stride + 1
```

and the same formula gives `W_out`. Integer division truncates. A non-positive result throws `std::invalid_argument` with message `spatial`. A rank other than 4, or a channel count other than `in_channels`, throws `Conv2d input shape`.

The convolution is cross-correlation, dilation 1, groups 1. The first call at a given spatial size allocates one workspace, at most 64 MiB. A larger request throws `conv workspace <bytes> exceeds 64 MiB budget`. A later shape that needs more memory frees and reallocates.

`backward` accumulates `grad_weight` and, when bias is on, `grad_bias`. The gradient's shape matches the forward output. A mismatch throws `Conv2d backward shape`.

### MaxPool2d

Header: `nn/layers/max_pool2d.hpp`

```cpp
MaxPool2d(Context& ctx, int kernel, int stride);
```

The window is square and the padding is 0. `kernel` and `stride` are at least 1. Output height and width use `conv_out_size` with pad 0. `forward` takes rank 4 and returns `(N, C, H_out, W_out)`. A rank other than 4 throws `MaxPool2d input rank`. A gradient whose shape differs from the forward output throws `MaxPool2d backward shape`. There are no parameters. `zero_grad` is empty.

### BatchNorm2d

Header: `nn/layers/batch_norm2d.hpp`

```cpp
BatchNorm2d(Context& ctx, int channels);
```

`channels` is at least 1 (`BatchNorm2d channels`). The layer owns scale and shift parameters and the running statistics.

| Tensor | Shape | Name | Initial value |
|---|---|---|---|
| weight (scale) | `(channels)` | `weight` | 1 |
| bias (shift) | `(channels)` | `bias` | 0 |
| running mean | `(channels)` | `running_mean` | 0 |
| running variance | `(channels)` | `running_var` | 1 |

The batch counter starts at 0. `parameters()` returns `weight` and `bias`. `buffers()` returns `running_mean` and `running_var`. Both buffers share the counter through `NamedBuffer::batches`.

`forward` takes `(N, channels, H, W)` and returns the same shape. A rank other than 4, or a channel mismatch, throws `BatchNorm2d input shape`. The mode is spatial and epsilon is `1e-5`.

In train mode the layer requires `N * H * W > 1`. A smaller count throws `BatchNorm train mode needs more than one value per channel`. Each train forward increments the counter, then updates the running statistics with factor `1 / batches`. Eval forward uses the stored mean and variance and leaves them unchanged.

`backward` runs in train mode and adds into the scale and shift gradients. Eval mode throws `BatchNorm backward requires train mode`. A missing forward or a mismatched gradient throws `BatchNorm2d backward shape`.

`set_training(true)` selects train mode. `set_training(false)` selects eval mode. `Network::train` and `Network::eval` call `set_training` on every layer.

### Flatten

Header: `nn/layers/flatten.hpp`

```cpp
class Flatten : public Layer { /* default constructor */ };
```

No context and no parameters. `forward` takes `(N, C, H, W)` and returns `(N, C*H*W)`, copying the values in NCHW order. A rank other than 4 throws `Flatten input rank`. A feature count that does not fit in a signed 32-bit integer throws `Flatten features out of range`.

`backward` takes `(N, C*H*W)` and returns `(N, C, H, W)`. Call `forward` first. A backward with no forward throws `Flatten backward without forward`. A mismatched gradient throws `Flatten backward shape`.

### GlobalAvgPool2d

Header: `nn/layers/global_avg_pool2d.hpp`

```cpp
explicit GlobalAvgPool2d(Context& ctx);
```

No parameters. `forward` takes `(N, C, H, W)` and returns `(N, C, 1, 1)`. A rank other than 4 throws `GlobalAvgPool2d input rank`. `backward` spreads the `(N, C, 1, 1)` gradient back across the spatial map. A mismatched gradient throws `GlobalAvgPool2d backward shape`.

### ResidualBlock

Header: `nn/layers/residual_block.hpp`

```cpp
ResidualBlock(Context& ctx, int in_channels, int out_channels, int stride, std::mt19937& rng);
```

`in_channels`, `out_channels`, and `stride` are at least 1 (`ResidualBlock size`). The block is a basic residual unit:

```
main = bn2(conv2(relu(bn1(conv1(x)))))
skip = x                                 when stride == 1 and in_channels == out_channels
skip = proj_bn(proj(x))                  otherwise
y    = relu(main + skip)
```

`conv1` is 3×3 with the given stride and pad 1. `conv2` is 3×3 with stride 1 and pad 1. `proj` is 1×1 with the given stride and pad 0. Every convolution has bias off. Each one consumes `rng` through `kaiming_normal`, in order `conv1`, `conv2`, then `proj` when the projection exists. Batch-norm scale, bias, and running statistics use the `BatchNorm2d` initial values and consume no random numbers.

`forward` takes `(N, in_channels, H, W)`. Output spatial size follows the 3×3 convolution with the block stride and pad 1. Output channels are `out_channels`.

Parameter order:

| Name | Present |
|---|---|
| `conv1.weight` | always |
| `bn1.weight`, `bn1.bias` | always |
| `conv2.weight` | always |
| `bn2.weight`, `bn2.bias` | always |
| `proj.weight` | when the skip projects |
| `proj_bn.weight`, `proj_bn.bias` | when the skip projects |

Buffers follow the same prefixes: `bn1.running_mean`, `bn1.running_var`, `bn2.running_mean`, `bn2.running_var`, and `proj_bn.running_mean`, `proj_bn.running_var` when the skip projects.

`set_training` forwards to every child. Train mode follows the `BatchNorm2d` rules on each norm in the block, including `N * H * W > 1` at that child's spatial size.

## Configuration

Header: `nn/config.hpp`

Builders call `validate` before they allocate. You can call it yourself. A failed check throws `std::invalid_argument` and the message is the field name.

```cpp
int conv_out_size(int in, int kernel, int stride, int pad);
```

The result is `(in + 2 * pad - kernel) / stride + 1`. `kernel` and `stride` are at least 1 (`kernel`, `stride`). `pad` is at least 0 (`pad`). A result below 1 throws `spatial`.

```cpp
enum class Activation : uint32_t { Relu = 1 };
```

### MlpConfig

```cpp
struct MlpConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
  Activation activation = Activation::Relu;
};
```

`input_size`, every hidden width, and `output_size` are at least 1. `activation` is `Activation::Relu`. An empty `hidden` builds one linear layer. `build_mlp` emits `Linear`, `ReLU`, `Linear`, `ReLU`, ..., `Linear`. The last linear layer has no ReLU. Names are `fc.0`, `fc.1`, ... Input shape is `(N, input_size)`. Output shape is `(N, output_size)`.

### ConvSpec and PoolSpec

```cpp
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
```

`ConvSpec` describes one square convolution. `PoolSpec` describes the square max pool, with pad 0, applied after a convolution.

### ConvConfig

```cpp
struct ConvConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  std::vector<int> fc;
};
```

`in_channels`, `in_height`, and `in_width` are at least 1. `convs` and `fc` are nonempty. Each `out_channels` and each fully connected width is at least 1. `validate` walks the spatial size through every convolution and the following pool, and rejects a map that would shrink below 1×1.

`build_conv` emits, for each entry in `convs`, `Conv2d` with bias on, `ReLU`, and `MaxPool2d`. Names are `conv.i` and `pool.i`. It then appends `Flatten` and `fc` linear layers named `fc.i`, with `ReLU` between linear layers. The first linear layer sees `channels * height * width` after the last pool.

### ResNetConfig

```cpp
struct ResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  int output_size = 0;
};
```

`in_channels`, `in_height`, `in_width`, `stem_channels`, and `output_size` are at least 1. `blocks_per_stage` and `stage_widths` have the same nonempty length. Each stage width is at least 1. Each stage has at least one block.

`build_resnet` emits:

1. `Conv2d` 3×3, stride 1, pad 1, bias off, named `stem`.
2. `BatchNorm2d` named `stem.bn`.
3. `ReLU`.
4. One `ResidualBlock` per block, named `stages.{s}.blocks.{b}`. Stage 0 uses stride 1. The first block of every later stage uses stride 2. The other blocks use stride 1.
5. `GlobalAvgPool2d`, `Flatten`, and `Linear` named `head`.

The head's input features are the last stage width. The stem keeps the input spatial size. A projection appears when stride is not 1 or the channel count changes.

### ConvResNetConfig

```cpp
struct ConvResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  std::vector<int> fc;
};
```

This config builds one image network with a conv+pool trunk, residual stages, or both. `fc` is always nonempty, and every width in it is at least 1.

| Choice | Rule |
|---|---|
| Trunk | `convs` nonempty. Each stage is `Conv2d` (bias on), `ReLU`, `MaxPool2d` using `pool`. |
| Residual stem | `stem_channels > 0` inserts the 3×3 stride-1 pad-1 bias-off stem, `BatchNorm2d`, and `ReLU`. This requires residual stages. |
| Residual stages | `blocks_per_stage` nonempty and the same length as `stage_widths`. Stage 0 uses stride 1. Later stages downsample on the first block. |
| Head | `fc` linear layers named `fc.i`, with `ReLU` between them. |

At least one of `convs` and `blocks_per_stage` is nonempty. `stem_channels < 0` throws `stem_channels`. When stages are present, the builder appends `GlobalAvgPool2d` and `Flatten`, and the first linear layer sees the last stage width. When stages are absent, it appends `Flatten`, and the first linear layer sees `channels * height * width` after the trunk. `pool` is stored in the checkpoint even when `convs` is empty. `validate` checks `pool` when the trunk is present.

Names match the other builders: `conv.i`, `pool.i`, `stem`, `stem.bn`, `stages.{s}.blocks.{b}`, and `fc.i`.

Random draws follow construction order. Each `conv.i.weight` is drawn, then `stem.weight` when the stem is present, then each block's `conv1.weight`, `conv2.weight`, and `proj.weight` when that block projects, then each `fc.i.weight`.

## Network

Header: `nn/network.hpp`

```cpp
enum class Family : uint32_t { Mlp = 1, Conv = 2, ResNet = 3, ConvResNet = 4 };

class Network {
 public:
  Network(Context& ctx, Family family, std::vector<std::unique_ptr<Layer>> layers,
          std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig> config);

  const Tensor& forward(const Tensor& input);
  const Tensor& backward(const Tensor& grad_output);
  void zero_grad();
  void train();
  void eval();
  bool training() const;
  Family family() const;
  const std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig>& config() const;

  std::vector<NamedParam> parameters() const;
  std::vector<NamedBuffer> buffers() const;

  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);
};

struct Network::Loaded {
  Network network;
  std::optional<Sgd> optimizer;
  uint64_t step = 0;
};
```

Header: `nn/builders.hpp`

```cpp
Network build_mlp(Context& ctx, const MlpConfig& config, uint32_t seed);
Network build_conv(Context& ctx, const ConvConfig& config, uint32_t seed);
Network build_resnet(Context& ctx, const ResNetConfig& config, uint32_t seed);
Network build_conv_resnet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
```

`Network` is movable and not copyable. The usual construction is a builder. Each builder validates the config, seeds one `std::mt19937`, and returns a network in train mode. `load` rebuilds with seed 0 and then overwrites every parameter from the file, so the seed used at load time leaves the restored weights unchanged.

`forward` runs the layers in order and returns the last layer's output. `backward` runs them in reverse and returns the gradient of the network input. An empty layer list throws `empty network`.

`train()` and `eval()` set the network flag and call `set_training` on every layer. `training()` reports that flag.

`zero_grad()` zeros every parameter gradient. It leaves parameters, velocity, and batch-norm buffers unchanged.

`config()` returns the variant stored at construction. Read it with `std::get<MlpConfig>`, `std::get<ConvConfig>`, `std::get<ResNetConfig>`, or `std::get<ConvResNetConfig>`.

The public constructor is for a caller who already owns the layers. `save` writes `family` and `config`. `load` rebuilds with the builder for that family, then requires the parameter and batch-norm records to match that builder's names and shapes. A checkpoint round-trip is defined for a network produced by one of the four builders.

## ConvResNet

Header: `nn/conv_resnet.hpp`

```cpp
class ConvResNet {
 public:
  ConvResNet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
  explicit ConvResNet(Network network);

  Network& network();
  const Network& network() const;

  const Tensor& forward(const Tensor& input);
  const Tensor& backward(const Tensor& grad_output);
  void zero_grad();
  void train();
  void eval();
  bool training() const;
  Family family() const;
  const ConvResNetConfig& config() const;

  std::vector<NamedParam> parameters() const;
  std::vector<NamedBuffer> buffers() const;
  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);
};

struct ConvResNet::Loaded {
  ConvResNet model;
  std::optional<Sgd> optimizer;
  uint64_t step = 0;
};
```

`ConvResNet` owns one `Network` of family `ConvResNet`. It is movable and not copyable. The config constructor calls `build_conv_resnet`. The `Network` constructor moves the network in and throws `ConvResNet family` when `family()` is anything else.

`forward`, `backward`, `zero_grad`, `train`, `eval`, `parameters`, `buffers`, and `save` forward to that network. `config()` returns the `ConvResNetConfig`. `network()` exposes the owned network when a function takes `Network&`, including `Sgd`'s interaction with `parameters()` (call `model.parameters()`; it returns the same tensors).

```cpp
nn::ConvResNet model(ctx, config, /*seed=*/1);
nn::Sgd optimizer(model.parameters(), 0.1f, 0.9f);

model.train();
model.zero_grad();
const nn::Tensor& logits = model.forward(images);
nn::LossResult loss = nn::softmax_cross_entropy(ctx, logits, labels);
model.backward(loss.grad_logits);
optimizer.step();

model.save(path, &optimizer);
nn::ConvResNet::Loaded loaded =
    nn::ConvResNet::load(ctx, path, nn::LoadMode::Resume);
```

`ConvResNet::load` uses `Network::load`, then requires family `ConvResNet`. A file of another family throws `bad checkpoint family`.

Three legal configs, each with `fc` set:

```cpp
// Trunk only. After the pool, Flatten feeds fc.
config.convs = {nn::ConvSpec{8, 3, 1, 1}};
config.pool = nn::PoolSpec{2, 2};
config.fc = {10};

// Residual stages only. Global average pool feeds fc.
config.stem_channels = 8;
config.blocks_per_stage = {1, 1};
config.stage_widths = {8, 16};
config.fc = {10};

// Trunk, then stem and residual stages, then fc.
config.convs = {nn::ConvSpec{8, 3, 1, 1}};
config.pool = nn::PoolSpec{2, 2};
config.stem_channels = 8;
config.blocks_per_stage = {1, 1};
config.stage_widths = {8, 16};
config.fc = {10};
```

## Sgd

Header: `nn/optim.hpp`

```cpp
class Sgd {
 public:
  Sgd(std::vector<NamedParam> params, float learning_rate, float momentum);

  void step();
  void set_learning_rate(float learning_rate);
  float learning_rate() const;
  float momentum() const;
  uint64_t step_count() const;
  void set_step_count(uint64_t step);

  const std::vector<Tensor>& velocity() const;
  std::vector<Tensor>& velocity();
  const std::vector<NamedParam>& params() const;
};
```

Construct `Sgd` from `network.parameters()` or `model.parameters()` after the network exists. An empty list throws `SGD has no parameters`. The optimizer is movable and not copyable. It copies the parameter pointers and allocates one velocity tensor per parameter on the first parameter's context. Velocity starts at zero. `step_count()` starts at 0.

`step()` applies Polyak momentum on the context stream, then increments `step_count` by one:

```
velocity = momentum * velocity + gradient
parameter = parameter - learning_rate * velocity
```

`set_learning_rate` changes the rate used by later steps. Momentum stays at the value given to the constructor. `set_step_count` sets the counter that `save` writes. `load` with `LoadMode::Resume` calls it with the stored step.

Pass the same optimizer to `save`. `save` checks pointer identity, in order, against the network's parameters. A mismatch throws `optimizer does not belong to this network`.

Moving a `Network` or a `ConvResNet` moves the `unique_ptr<Layer>` elements. The layer objects stay where they are, so an `Sgd` built from those parameters remains valid after the move.

## Checkpoints

Header: `nn/checkpoint.hpp`

```cpp
enum class LoadMode { Inference, Resume };
```

```cpp
network.save(path, &optimizer);  // parameters, batch-norm state, and momentum
network.save(path, nullptr);     // parameters and batch-norm state; the stored step is 0

nn::Network::Loaded loaded = nn::Network::load(ctx, path, nn::LoadMode::Inference);
nn::Network::Loaded resumed = nn::Network::load(ctx, path, nn::LoadMode::Resume);
```

`save` synchronizes, copies device state to the host, and writes `path` with a `.tmp` suffix. It `fsync`s, closes, and renames that file onto `path`. A failure removes the temporary file and leaves an existing `path` in place. Host file errors throw `checkpoint write: ` plus `strerror`.

`Loaded::network` is the rebuilt network. `Loaded::step` is the step stored in the file. `Loaded::optimizer` is set for `LoadMode::Resume`. `LoadMode::Inference` parses an optimizer block when the file has one, checks it, and leaves `optimizer` empty. `LoadMode::Resume` throws `optimizer block missing` when the file has no optimizer block. The restored network is in the train or eval mode stored in the file.

`load` checks every record before it copies any parameter to the device. A file that fails validation leaves the caller with the exception and no half-updated network.

| Condition | Message |
|---|---|
| Unreadable, shorter than 32 bytes, or cut off mid-record | `truncated checkpoint` |
| Wrong magic | `bad checkpoint magic` |
| Version other than 1 | `unsupported checkpoint version N` |
| Bad train flag, optimizer flag, or reserved word | `bad checkpoint header` |
| Family outside 1..4 | `bad checkpoint family` |
| `ConvResNet::load` of another family | `bad checkpoint family` |
| Parameter or batch-norm record disagrees with the rebuilt network | `checkpoint tensor mismatch: ...` |
| `Resume` and the optimizer flag is 0 | `optimizer block missing` |
| Optimizer record count disagrees | `optimizer state mismatch` |
| `save` optimizer belongs to other parameters | `optimizer does not belong to this network` |

The byte layout is version 1, little-endian, in [README.md](README.md). The file holds the config, the parameters, the batch-norm running statistics, and, when requested, the velocity. It leaves out activation caches, gradients, workspaces, and cuDNN descriptors.

## kaiming_normal

Header: `nn/init.hpp`

```cpp
void kaiming_normal(Tensor& weight, int fan_in, std::mt19937& rng);
```

Fills `weight` with a normal distribution of mean 0 and standard deviation `sqrt(2 / fan_in)`, then synchronizes. `fan_in` is at least 1 (`kaiming fan_in`). `Linear` uses `in_features`. `Conv2d` uses `in_channels * kernel * kernel`. Builders call this for you. Call it when you construct a `Linear` or `Conv2d` yourself and want to draw again into an existing weight tensor.
