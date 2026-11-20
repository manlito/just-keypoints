#include "models/xfeat/xfeat.h"

#include <stdexcept>

namespace jkp {

Xfeat::Conv Xfeat::conv(const Weights& weights, const std::string& name) const {
  return {&weights.get(name + ".weight"), &weights.get(name + ".bias")};
}

Xfeat::Xfeat(const Weights& weights) {
  skip_ = conv(weights, "skip");
  auto blocks = [&](Conv* out, int n, const std::string& name) {
    for (int i = 0; i < n; ++i) out[i] = conv(weights, name + "." + std::to_string(i));
  };
  blocks(block1_, 4, "block1");
  blocks(block2_, 2, "block2");
  blocks(block3_, 3, "block3");
  blocks(block4_, 3, "block4");
  blocks(block5_, 4, "block5");
  blocks(fusion_, 3, "fusion");
  blocks(heatmap_, 3, "heatmap");
  blocks(keypoint_, 4, "keypoint");
  if (weights.meta_int("dim") != 64 || keypoint_[3].w->shape[0] != 65)
    throw std::runtime_error("xfeat: unexpected weights");
  detect_.top_k = weights.meta_int("top_k");
  detect_.threshold = std::stof(weights.meta("detection_threshold"));
  detect_.window = weights.meta_int("nms");
}

void Xfeat::extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels, int top_k,
                    Features* out, const StageSink& sink) const {
  // The network sees the image resized down to a multiple of 32, as the original.
  const int H = height / 32 * 32, W = width / 32 * 32;
  if (H == 0 || W == 0) throw std::runtime_error("xfeat needs an image of at least 32 x 32 pixels");
  auto emit = [&](const char* name, const TensorPtr& t) {
    if (sink) sink(name, t);
  };
  auto c1 = [&](const TensorPtr& t, const Conv& c, Act act = Act::Relu) { return ops.conv2d(t, *c.w, c.b, act); };
  auto c2 = [&](const TensorPtr& t, const Conv& c) { return ops.conv2d_stride2(t, *c.w, c.b, Act::Relu); };

  TensorPtr x = ops.image(data, width, height, stride, channels);
  if (H != height || W != width) x = ops.resize(x, H, W);
  emit("resized", x);
  TensorPtr xn = ops.gray_instance_norm(x);
  x = nullptr;
  emit("norm", xn);

  TensorPtr x1 = c2(c1(c2(c1(xn, block1_[0]), block1_[1]), block1_[2]), block1_[3]);
  TensorPtr skip = c1(ops.avg_pool(xn, 4), skip_, Act::None);
  emit("x1", x1), emit("skip", skip);
  TensorPtr x2 = c1(c1(ops.add(x1, skip, Act::None), block2_[0]), block2_[1]);
  x1 = skip = nullptr;
  TensorPtr x3 = c1(c1(c2(x2, block3_[0]), block3_[1]), block3_[2]);
  TensorPtr x4 = c1(c1(c2(x3, block4_[0]), block4_[1]), block4_[2]);
  TensorPtr x5 = c1(c1(c1(c2(x4, block5_[0]), block5_[1]), block5_[2]), block5_[3]);
  emit("x2", x2), emit("x3", x3), emit("x4", x4), emit("x5", x5);
  x2 = nullptr;

  // Pyramid fusion at 1/8 resolution.
  TensorPtr x4u = ops.resize(x4, x3->h, x3->w), x5u = ops.resize(x5, x3->h, x3->w);
  emit("x4u", x4u), emit("x5u", x5u);
  x4 = x5 = nullptr;
  TensorPtr f = ops.add(ops.add(x3, x4u, Act::None), x5u, Act::None);
  x3 = x4u = x5u = nullptr;
  TensorPtr feats = c1(c1(c1(f, fusion_[0]), fusion_[1]), fusion_[2], Act::None);
  f = nullptr;
  emit("feats", feats);

  // Heads: reliability from the features, keypoint logits from the 8x8 cells of the normalized image.
  TensorPtr heatmap = c1(c1(c1(feats, heatmap_[0]), heatmap_[1]), heatmap_[2], Act::Sigmoid);
  emit("heatmap", heatmap);
  TensorPtr unfolded = ops.space_to_depth(xn, 8);
  xn = nullptr;
  emit("unfolded", unfolded);
  TensorPtr logits = c1(c1(c1(c1(unfolded, keypoint_[0]), keypoint_[1]), keypoint_[2]), keypoint_[3], Act::None);
  unfolded = nullptr;
  emit("logits", logits);
  TensorPtr feats_n = ops.l2_normalize(feats);
  feats = nullptr;
  emit("feats_n", feats_n);

  XfeatDetectParams params = detect_;
  if (top_k > 0) params.top_k = top_k;
  XfeatKeypoints kpts;
  ops.detect_xfeat(logits, heatmap, feats_n, params, &kpts);

  // Pixel i of the resized image is the point (i + 0.5) * r - 0.5 of the original (r = original / resized).
  const double rx = static_cast<double>(width) / W, ry = static_cast<double>(height) / H;
  out->n = static_cast<int>(kpts.score.size());
  out->dim = kpts.dim;
  out->xy.resize(2 * static_cast<size_t>(out->n));
  for (int i = 0; i < out->n; ++i) {
    out->xy[2 * i] = static_cast<float>((kpts.pix[2 * i] + 0.5) * rx - 0.5);
    out->xy[2 * i + 1] = static_cast<float>((kpts.pix[2 * i + 1] + 0.5) * ry - 0.5);
  }
  out->score = std::move(kpts.score);
  out->desc = std::move(kpts.desc);
  out->rank.clear();
  out->cov.clear();
}

}  // namespace jkp
