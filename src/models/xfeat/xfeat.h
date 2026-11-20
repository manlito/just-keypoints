// XFeat graph (sparse mode: keypoints and 64-dim descriptors), written against the backend-independent Ops
// interface. Mirrors tools/xfeat_folded.py, which is the executable specification.
#pragma once

#include <functional>
#include <string>

#include "core/features.h"
#include "core/ops.h"
#include "core/weights.h"

namespace jkp {

class Xfeat {
 public:
  explicit Xfeat(const Weights& weights);

  // Called with every intermediate tensor when set (golden tests).
  using StageSink = std::function<void(const char* name, const TensorPtr& tensor)>;

  // data: 8-bit image with 1, 3 or 4 channels, at least 32 x 32. top_k <= 0: the default stored in the
  // weights file (4096).
  void extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels, int top_k,
               Features* out, const StageSink& sink = nullptr) const;

  int descriptor_dim() const { return 64; }

 private:
  struct Conv {
    const Param* w = nullptr;
    const Param* b = nullptr;
  };
  Conv conv(const Weights& weights, const std::string& name) const;

  Conv skip_, block1_[4], block2_[2], block3_[3], block4_[3], block5_[4], fusion_[3], heatmap_[3], keypoint_[4];
  XfeatDetectParams detect_;
};

}  // namespace jkp
