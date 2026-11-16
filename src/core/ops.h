// Backend interface. A model graph (src/models/*) is written once against Ops; each backend
// (cpu, cuda, metal, webgpu) implements these operations on its own tensor storage.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/weights.h"

namespace jkp {

// A CxHxW float tensor owned by a backend.
struct Tensor {
  int c = 0, h = 0, w = 0;
  virtual ~Tensor() = default;
  // Copies the values to host memory in CHW order (dst holds c*h*w floats). Used by tests and tools.
  virtual void read(float* dst) const = 0;
};
using TensorPtr = std::shared_ptr<Tensor>;

enum class Act { None, Selu, Sigmoid };

struct DetectParams {
  int top_k = -1;          // > 0: top-k mode, otherwise threshold mode
  float scores_th = 0.2f;  // threshold mode: keep nms score > scores_th (falls back to the mean score)
  int n_limit = 5000;      // threshold mode: cap, best first when exceeded
};

struct Keypoints {
  std::vector<float> xy_norm;  // [n][2] in [-1, 1], as produced by the detector
  std::vector<float> score;    // [n]
  int n() const { return static_cast<int>(score.size()); }
};

// Weights of the sparse deformable descriptor head.
struct SddhParams {
  int kernel = 3;       // patch size K
  int positions = 16;   // sample positions M
  const Param *offset0_w, *offset0_b;  // [2M, C, K, K], [2M]
  const Param *offset1_w, *offset1_b;  // [2M, 2M, 1, 1], [2M]
  const Param* sf_w;                   // [C, C, 1, 1]
  const Param* agg_w;                  // [M, C, D]
};

class Ops {
 public:
  virtual ~Ops() = default;

  // 8-bit image (1, 3 or 4 channels, alpha ignored) -> 3xHxW in [0, 1].
  virtual TensorPtr image(const uint8_t* data, int width, int height, int stride, int channels) = 0;
  virtual TensorPtr pad_replicate(const TensorPtr& x, int left, int right, int top, int bottom) = 0;
  virtual TensorPtr crop(const TensorPtr& x, int left, int top, int width, int height) = 0;
  // Stride 1, "same" zero padding. weight: [Cout, Cin, k, k] with k odd; bias may be null.
  virtual TensorPtr conv2d(const TensorPtr& x, const Param& weight, const Param* bias, Act act) = 0;
  // 3x3 deformable convolution. Offsets come from a 3x3 convolution on x, clamped to +-max(h, w) / 4.
  virtual TensorPtr deform_conv2d(const TensorPtr& x, const Param& offset_weight, const Param& offset_bias,
                                  const Param& weight, const Param& bias, Act act) = 0;
  virtual TensorPtr add(const TensorPtr& a, const TensorPtr& b, Act act) = 0;
  virtual TensorPtr avg_pool(const TensorPtr& x, int k) = 0;
  // Bilinear, align_corners = true.
  virtual TensorPtr upsample(const TensorPtr& x, int scale) = 0;
  virtual TensorPtr concat(const std::vector<TensorPtr>& xs) = 0;
  // L2 normalization across channels.
  virtual TensorPtr l2_normalize(const TensorPtr& x) = 0;

  // Fused forms of sequences in the graph. The defaults run the separate operations; a backend overrides
  // them to save the intermediate tensors, with the same results.
  // concat of xs[i] upsampled by scales[i] (1 = as is).
  virtual TensorPtr upsample_concat(const std::vector<TensorPtr>& xs, const std::vector<int>& scales) {
    std::vector<TensorPtr> parts;
    for (size_t i = 0; i < xs.size(); ++i) parts.push_back(scales[i] == 1 ? xs[i] : upsample(xs[i], scales[i]));
    return concat(parts);
  }
  // crop(l2_normalize(x), ...).
  virtual TensorPtr l2_normalize_crop(const TensorPtr& x, int left, int top, int width, int height) {
    return crop(l2_normalize(x), left, top, width, height);
  }

  // Keypoint detection on a 1xHxW score map: NMS, selection, soft-argmax refinement.
  virtual void detect(const TensorPtr& score_map, const DetectParams& params, Keypoints* out) = 0;
  // Descriptors [n][D] for the given keypoints from a CxHxW feature map.
  virtual void describe(const TensorPtr& feature_map, const Keypoints& kpts, const SddhParams& params,
                        std::vector<float>* desc) = 0;
};

std::unique_ptr<Ops> make_cpu_ops(int num_threads);
// Only in builds with the CUDA backend (JKP_HAVE_CUDA). device: GPU index.
std::unique_ptr<Ops> make_cuda_ops(int device);
// Only in builds with the Metal backend (JKP_HAVE_METAL). device: index in MTLCopyAllDevices().
std::unique_ptr<Ops> make_metal_ops(int device);

}  // namespace jkp
