#include "nn/transformer.hpp"

#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"
#include "nn/layers/layer_norm.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/transformer_block.hpp"

#include <random>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace nn {
namespace {

std::string join_name(const std::string& prefix, const std::string& name) {
  if (prefix.empty()) return name;
  return prefix + "." + name;
}

const TransformerConfig& transformer_config(const Network& network) {
  return std::get<TransformerConfig>(network.config());
}

}  // namespace

Transformer::Transformer(Context& ctx, const TransformerConfig& config, uint32_t seed) : ctx_(&ctx) {
  validate(config);
  std::mt19937 rng(seed);
  tok_emb_ = std::make_unique<Embedding>(ctx, config.vocab_size, config.d_model, rng);
  tok_emb_->set_name("tok_emb");
  if (config.positional == PositionalEncoding::LearnedAbsolute) {
    pos_emb_ = std::make_unique<Embedding>(ctx, config.max_seq_len, config.d_model, rng);
    pos_emb_->set_name("pos_emb");
  }
  TransformerBlockSpec block_spec;
  block_spec.d_model = config.d_model;
  block_spec.num_heads = config.num_heads;
  block_spec.d_ff = config.d_ff;
  block_spec.dropout = config.dropout;
  block_spec.bias = config.bias;
  block_spec.activation = config.activation;
  block_spec.use_rope = false;

  std::vector<std::unique_ptr<Layer>> layers;
  for (int i = 0; i < config.num_layers; ++i) {
    auto block = std::make_unique<TransformerBlock>(ctx, block_spec, rng);
    block->set_name("blocks." + std::to_string(i));
    block_ptrs_.push_back(block.get());  // before the move
    layers.push_back(std::move(block));
  }
  auto final_ln = std::make_unique<LayerNorm>(ctx, config.d_model);
  final_ln->set_name("final_ln");
  final_ln_ptr_ = final_ln.get();
  layers.push_back(std::move(final_ln));
  auto lm_head = std::make_unique<Linear>(ctx, config.d_model, config.vocab_size, config.bias, rng);
  lm_head->set_name("lm_head");
  lm_head_ptr_ = lm_head.get();
  layers.push_back(std::move(lm_head));
  network_ = Network(ctx, Family::Transformer, std::move(layers), config);
}

const Tensor& Transformer::forward(const IndexTensor& tokens) {
  if (tokens.shape().size() != 2) throw std::invalid_argument("tokens");
  const int n = tokens.shape()[0];
  const int t = tokens.shape()[1];
  if (t > transformer_config(*network_).max_seq_len) throw std::invalid_argument("max_seq_len");

  const Tensor& tok = tok_emb_->forward_indices(tokens);
  if (!pos_ids_ || pos_ids_->shape()[0] != n || pos_ids_->shape()[1] != t) {
    pos_ids_.emplace(*ctx_, std::vector<int>{n, t});
    std::vector<int64_t> host(static_cast<size_t>(pos_ids_->numel()));
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < t; ++col) {
        host[static_cast<size_t>(row) * static_cast<size_t>(t) + static_cast<size_t>(col)] = col;
      }
    }
    pos_ids_->copy_from_host(host.data(), pos_ids_->numel());
    // copy_from_host is async; the local buffer has to stay alive until it finishes.
    ctx_->synchronize();
  }
  const Tensor& pos = pos_emb_->forward_indices(*pos_ids_);
  ensure_tensor(embedded_, *ctx_, tok.shape());
  launch_add2(stream_of(*ctx_), embedded_->data(), tok.data(), pos.data(), tok.numel());
  const Tensor& logits = network_->forward(*embedded_);
  forwarded_ = true;
  used_embeddings_ = true;
  return logits;
}

const Tensor& Transformer::forward(const Tensor& embeddings) {
  const TransformerConfig& config = transformer_config(*network_);
  if (embeddings.rank() != 3 || embeddings.shape().back() != config.d_model) {
    throw std::invalid_argument("embeddings");
  }
  const Tensor& logits = network_->forward(embeddings);
  forwarded_ = true;
  used_embeddings_ = false;
  return logits;
}

const Tensor& Transformer::backward(const Tensor& grad_logits) {
  if (!forwarded_) throw Error("Transformer backward without forward");
  const Tensor& grad_x = network_->backward(grad_logits);
  if (used_embeddings_) {
    tok_emb_->backward(grad_x);
    pos_emb_->backward(grad_x);
  }
  return grad_x;
}

void Transformer::zero_grad() {
  tok_emb_->zero_grad();
  if (pos_emb_) pos_emb_->zero_grad();
  network_->zero_grad();
}

void Transformer::train() {
  tok_emb_->set_training(true);
  if (pos_emb_) pos_emb_->set_training(true);
  network_->train();
}

void Transformer::eval() {
  tok_emb_->set_training(false);
  if (pos_emb_) pos_emb_->set_training(false);
  network_->eval();
}

bool Transformer::training() const { return network_->training(); }

std::vector<NamedParam> Transformer::parameters() const {
  std::vector<NamedParam> params;
  for (NamedParam param : tok_emb_->parameters()) {
    param.name = join_name(tok_emb_->name(), param.name);
    params.push_back(param);
  }
  if (pos_emb_) {
    for (NamedParam param : pos_emb_->parameters()) {
      param.name = join_name(pos_emb_->name(), param.name);
      params.push_back(param);
    }
  }
  const std::vector<NamedParam> trunk = network_->parameters();
  params.insert(params.end(), trunk.begin(), trunk.end());
  return params;
}

const Network& Transformer::network() const { return *network_; }

}  // namespace nn
