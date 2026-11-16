#include "models/raco/raco.h"

#include <stdexcept>

namespace jkp {

Raco::Conv Raco::conv(const Weights& weights, const std::string& name) const {
  return {&weights.get(name + ".weight"), weights.find(name + ".bias")};
}

Raco::Block Raco::block(const Weights& weights, const std::string& name) const {
  Block b;
  b.conv1 = conv(weights, name + ".conv1");
  b.conv2 = conv(weights, name + ".conv2");
  if (weights.find(name + ".down.weight")) b.down = conv(weights, name + ".down");
  return b;
}

Raco::Raco(const Weights& weights) {
  mean_ = &weights.get("norm.mean");
  std_ = &weights.get("norm.std");
  if (mean_->size() != 3 || std_->size() != 3) throw std::runtime_error("raco: bad normalization constants");
  for (int i = 0; i < 4; ++i) {
    block_[i] = block(weights, "block" + std::to_string(i + 1));
    agg_[i] = conv(weights, "conv" + std::to_string(i + 1));
    score_[i] = conv(weights, "score_head." + std::to_string(2 * i));
    cov_[i] = conv(weights, "cov." + std::to_string(i));
  }
  for (int i = 0; i < weights.meta_int("ranker_blocks"); ++i)
    ranker_.push_back(block(weights, "ranker." + std::to_string(i)));
  ranker_out_ = conv(weights, "ranker.out");
  detect_.top_k = weights.meta_int("top_k");
  detect_.window = weights.meta_int("nms");
  detect_.temperature = std::stof(weights.meta("subpixel_temp"));
}

TensorPtr Raco::apply(Ops& ops, const TensorPtr& x, const Block& b) const {
  TensorPtr y = ops.conv2d(x, *b.conv1.w, b.conv1.b, Act::Selu);
  if (!b.down.w) return ops.conv2d(y, *b.conv2.w, b.conv2.b, Act::Selu);
  y = ops.conv2d(y, *b.conv2.w, b.conv2.b, Act::None);
  return ops.add(y, ops.conv2d(x, *b.down.w, b.down.b, Act::None), Act::Selu);
}

void Raco::extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
                   const RacoOptions& options, Features* out, const StageSink& sink) const {
  if (width < 1 || height < 1) throw std::runtime_error("image is empty");
  auto emit = [&](const char* name, const TensorPtr& t) {
    if (sink) sink(name, t);
  };
  auto conv2d = [&](const TensorPtr& t, const Conv& c, Act act) { return ops.conv2d(t, *c.w, c.b, act); };

  // ImageNet normalization, then pad to a multiple of 32, centred, replicating the border.
  int pw = (32 - width % 32) % 32, ph = (32 - height % 32) % 32;
  int left = pw / 2, top = ph / 2;
  TensorPtr x = ops.normalize(ops.image(data, width, height, stride, channels), mean_->data, std_->data);
  x = ops.pad_replicate(x, left, pw - left, top, ph - top);
  emit("x", x);

  // Encoder: full, 1/2, 1/8 and 1/32 resolution.
  TensorPtr x1 = apply(ops, x, block_[0]);
  TensorPtr x2 = apply(ops, ops.avg_pool(x1, 2), block_[1]);
  TensorPtr x3 = apply(ops, ops.avg_pool(x2, 4), block_[2]);
  TensorPtr x4 = apply(ops, ops.avg_pool(x3, 4), block_[3]);
  emit("x1", x1), emit("x2", x2), emit("x3", x3), emit("x4", x4);

  // Aggregation at full resolution.
  TensorPtr a1 = conv2d(x1, agg_[0], Act::Selu);
  TensorPtr a2 = conv2d(x2, agg_[1], Act::Selu);
  TensorPtr a3 = conv2d(x3, agg_[2], Act::Selu);
  TensorPtr a4 = conv2d(x4, agg_[3], Act::Selu);
  emit("a1", a1), emit("a2", a2), emit("a3", a3), emit("a4", a4);
  x1 = x2 = x3 = x4 = nullptr;
  TensorPtr x1234 = ops.concat({a1, ops.upsample(a2, 2), ops.upsample(a3, 8), ops.upsample(a4, 32)});
  a1 = a2 = a3 = a4 = nullptr;
  emit("x1234", x1234);

  // Score head: raw logits. The detector turns them into probabilities over the whole image.
  TensorPtr s = conv2d(x1234, score_[0], Act::Selu);
  s = conv2d(s, score_[1], Act::Selu);
  s = conv2d(s, score_[2], Act::Selu);
  TensorPtr logits = ops.crop(conv2d(s, score_[3], Act::None), left, top, width, height);
  emit("logits", logits);

  // Ranker head: residual blocks on the normalized image, then a 5x5 convolution with reflect padding.
  // A convolution over the reflect-padded tensor, cropped by the padding, is that convolution.
  TensorPtr ranker_map;
  if (options.ranker) {
    TensorPtr r = x;
    for (const Block& b : ranker_) r = apply(ops, r, b);
    int p = ranker_out_.w->shape[2] / 2;
    r = conv2d(ops.pad_reflect(r, p), ranker_out_, Act::None);
    ranker_map = ops.crop(r, left + p, top + p, width, height);
    emit("ranker_map", ranker_map);
  }
  x = nullptr;

  // Covariance head: Cholesky elements (L11, L21, L22) per pixel. The detector applies softplus to the
  // diagonal elements where it samples them.
  TensorPtr cholesky_raw;
  if (options.covariance) {
    TensorPtr c = x1234;
    for (int i = 0; i < 3; ++i)
      c = ops.crop(conv2d(ops.pad_reflect(c, 1), cov_[i], Act::LeakyRelu), 1, 1, c->w, c->h);
    cholesky_raw = ops.crop(conv2d(c, cov_[3], Act::None), left, top, width, height);
    emit("cholesky_raw", cholesky_raw);
  }
  x1234 = nullptr;

  RacoDetectParams params = detect_;
  if (options.top_k > 0) params.top_k = options.top_k;
  RacoKeypoints kpts;
  ops.detect_raco(logits, ranker_map, cholesky_raw, params, &kpts);

  out->n = static_cast<int>(kpts.score.size());
  out->dim = 0;
  out->xy = std::move(kpts.xy);
  out->score = std::move(kpts.score);
  out->rank = std::move(kpts.rank);
  out->desc.clear();
  // Sigma = L L^T with L = [[L11, 0], [L21, L22]].
  out->cov.resize(kpts.cholesky.size() / 3 * 4);
  for (size_t i = 0; i < kpts.cholesky.size() / 3; ++i) {
    const float l11 = kpts.cholesky[3 * i], l21 = kpts.cholesky[3 * i + 1], l22 = kpts.cholesky[3 * i + 2];
    float* c = &out->cov[4 * i];
    c[0] = l11 * l11;
    c[1] = c[2] = l11 * l21;
    c[3] = l21 * l21 + l22 * l22;
  }
}

}  // namespace jkp
