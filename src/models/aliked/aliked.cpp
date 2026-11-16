#include "models/aliked/aliked.h"

#include <stdexcept>
#include <string>

namespace jkp {

Aliked::Conv Aliked::conv(const Weights& weights, const std::string& name) const {
  Conv c;
  c.w = &weights.get(name + ".weight");
  c.b = weights.find(name + ".bias");
  c.off_w = weights.find(name + ".offset.weight");
  c.off_b = weights.find(name + ".offset.bias");
  return c;
}

Aliked::Aliked(const Weights& weights) {
  for (int i = 0; i < 4; ++i) {
    std::string b = "block" + std::to_string(i + 1);
    conv1_[i] = conv(weights, b + ".conv1");
    conv2_[i] = conv(weights, b + ".conv2");
    if (i > 0) down_[i] = conv(weights, b + ".down");
    agg_[i] = conv(weights, "conv" + std::to_string(i + 1));
    score_[i] = conv(weights, "score_head." + std::to_string(2 * i));
  }
  dim_ = weights.meta_int("dim");
  sddh_.kernel = weights.meta_int("K");
  sddh_.positions = weights.meta_int("M");
  sddh_.offset0_w = &weights.get("desc_head.offset_conv.0.weight");
  sddh_.offset0_b = &weights.get("desc_head.offset_conv.0.bias");
  sddh_.offset1_w = &weights.get("desc_head.offset_conv.2.weight");
  sddh_.offset1_b = &weights.get("desc_head.offset_conv.2.bias");
  sddh_.sf_w = &weights.get("desc_head.sf_conv.weight");
  sddh_.agg_w = &weights.get("desc_head.agg_weights");
}

TensorPtr Aliked::apply(Ops& ops, const TensorPtr& x, const Conv& c, Act act) const {
  if (c.off_w) return ops.deform_conv2d(x, *c.off_w, *c.off_b, *c.w, *c.b, act);
  return ops.conv2d(x, *c.w, c.b, act);
}

TensorPtr Aliked::block(Ops& ops, const TensorPtr& x, int i) const {
  TensorPtr y = apply(ops, x, conv1_[i], Act::Selu);
  if (i == 0) return apply(ops, y, conv2_[i], Act::Selu);
  y = apply(ops, y, conv2_[i], Act::None);
  return ops.add(y, apply(ops, x, down_[i], Act::None), Act::Selu);
}

void Aliked::dense(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
                   const StageSink& sink, TensorPtr* feature_map_out, TensorPtr* score_map_out) const {
  if (width < 1 || height < 1) throw std::runtime_error("image is empty");
  auto emit = [&](const char* name, const TensorPtr& t) {
    if (sink) sink(name, t);
  };

  // Pad to a multiple of 32, centred, replicating the border.
  int pw = (32 - width % 32) % 32, ph = (32 - height % 32) % 32;
  int left = pw / 2, top = ph / 2;
  TensorPtr x = ops.pad_replicate(ops.image(data, width, height, stride, channels), left, pw - left, top, ph - top);

  // Encoder: full, 1/2, 1/8 and 1/32 resolution.
  TensorPtr x1 = block(ops, x, 0);
  TensorPtr x2 = block(ops, ops.avg_pool(x1, 2), 1);
  TensorPtr x3 = block(ops, ops.avg_pool(x2, 4), 2);
  TensorPtr x4 = block(ops, ops.avg_pool(x3, 4), 3);
  emit("x1", x1), emit("x2", x2), emit("x3", x3), emit("x4", x4);

  // Aggregation at full resolution.
  TensorPtr a1 = apply(ops, x1, agg_[0], Act::Selu);
  TensorPtr a2 = apply(ops, x2, agg_[1], Act::Selu);
  TensorPtr a3 = apply(ops, x3, agg_[2], Act::Selu);
  TensorPtr a4 = apply(ops, x4, agg_[3], Act::Selu);
  emit("a1", a1), emit("a2", a2), emit("a3", a3), emit("a4", a4);
  TensorPtr x1234 = ops.upsample_concat({a1, a2, a3, a4}, {1, 2, 8, 32});
  emit("x1234", x1234);

  // Score head.
  if (score_map_out) {
    TensorPtr s = apply(ops, x1234, score_[0], Act::Selu);
    s = apply(ops, s, score_[1], Act::Selu);
    s = apply(ops, s, score_[2], Act::Selu);
    *score_map_out = ops.crop(apply(ops, s, score_[3], Act::Sigmoid), left, top, width, height);
    emit("score_map", *score_map_out);
  }
  *feature_map_out = ops.l2_normalize_crop(x1234, left, top, width, height);
  emit("feature_map", *feature_map_out);
}

void Aliked::describe_at(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
                         const float* xy, int n, std::vector<float>* desc) const {
  TensorPtr feature_map;
  dense(ops, data, width, height, stride, channels, nullptr, &feature_map, nullptr);
  // Same normalization as ALIKED.describe in cvg/LightGlue; the descriptor head maps it back to pixels.
  Keypoints kpts;
  kpts.xy_norm.resize(2 * static_cast<size_t>(n));
  kpts.score.assign(n, 0.0f);
  const float wh[2] = {static_cast<float>(width - 1), static_cast<float>(height - 1)};
  for (int i = 0; i < 2 * n; ++i) kpts.xy_norm[i] = 2.0f * xy[i] / wh[i % 2] - 1.0f;
  ops.describe(feature_map, kpts, sddh_, desc);
}

void Aliked::extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
                     const DetectParams& detect, Features* out, const StageSink& sink) const {
  TensorPtr feature_map, score_map;
  dense(ops, data, width, height, stride, channels, sink, &feature_map, &score_map);

  Keypoints kpts;
  ops.detect(score_map, detect, &kpts);
  ops.describe(feature_map, kpts, sddh_, &out->desc);

  out->n = kpts.n();
  out->dim = dim_;
  out->score = kpts.score;
  out->rank.clear();
  out->cov.clear();
  out->xy.resize(2 * static_cast<size_t>(out->n));
  const float wh[2] = {static_cast<float>(width - 1), static_cast<float>(height - 1)};
  for (int i = 0; i < 2 * out->n; ++i) out->xy[i] = wh[i % 2] * (kpts.xy_norm[i] + 1.0f) / 2.0f;
}

}  // namespace jkp
