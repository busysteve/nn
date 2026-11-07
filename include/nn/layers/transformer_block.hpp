#pragma once

#include "nn/layer.hpp"
#include "nn/layers/multi_head_attention.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

namespace nn {

class LayerNorm;
class Dropout;
class Linear;

enum class TransformerActivation : uint32_t { Gelu = 1, SwiGLU = 2 };

struct TransformerBlockSpec {
  int d_model = 0;
  int num_heads = 0;
  int d_ff = 0;           // width AFTER the gate. SwiGLU's first linear writes 2*d_ff.
  float dropout = 0.f;    // in [0, 1)
  bool bias = false;
  bool use_rope = false;  // pass through to MultiHeadAttention. true throws today.
  TransformerActivation activation = TransformerActivation::SwiGLU;
};

class TransformerBlock : public Layer {
 public:
  TransformerBlock(Context& ctx, const TransformerBlockSpec& spec, std::mt19937& rng);
  ~TransformerBlock() override;
  const Tensor& forward(const Tensor& input) override;  // (N,T,D)
  const Tensor& backward(const Tensor& grad_output) override;
  std::vector<NamedParam> parameters() const override;
  void zero_grad() override;
  void set_training(bool training) override;
  MultiHeadAttention& attn() { return *mha_; }
  const MultiHeadAttention& attn() const { return *mha_; }

 private:
  Context* ctx_;
  int d_model_;
  std::unique_ptr<LayerNorm> ln1_;
  std::unique_ptr<MultiHeadAttention> mha_;
  std::unique_ptr<Dropout> drop1_;
  std::unique_ptr<LayerNorm> ln2_;
  std::unique_ptr<Linear> fc1_;
  std::unique_ptr<Layer> act_;
  std::unique_ptr<Linear> fc2_;
  std::unique_ptr<Dropout> drop2_;
  std::optional<Tensor> hidden_;
  std::optional<Tensor> output_;
  std::optional<Tensor> grad_hidden_;
  std::optional<Tensor> grad_input_;
};

}  // namespace nn
