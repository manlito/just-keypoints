// ALIKED graph, written against the backend-independent Ops interface.
// Mirrors tools/folded.py, which is the executable specification.
#pragma once

#include <functional>
#include <vector>

#include "core/features.h"
#include "core/ops.h"
#include "core/weights.h"

namespace jkp {

class Aliked {
 public:
  explicit Aliked(const Weights& weights);

  // Called with every intermediate tensor when set (golden tests).
  using StageSink = std::function<void(const char* name, const TensorPtr& tensor)>;

  // data: 8-bit image with 1, 3 or 4 channels.
  void extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
               const DetectParams& detect, Features* out, const StageSink& sink = nullptr) const;

  // Descriptors [n][dim] at given keypoints, e.g. from another detector. xy: [n][2] pixels,
  // (0, 0) = centre of the top-left pixel. Runs the dense network; the score head is skipped.
  void describe_at(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels, const float* xy,
                   int n, std::vector<float>* desc) const;

  int descriptor_dim() const { return dim_; }

 private:
  struct Conv {
    const Param* w = nullptr;
    const Param* b = nullptr;       // may be null
    const Param* off_w = nullptr;   // set for deformable convolutions
    const Param* off_b = nullptr;
  };
  Conv conv(const Weights& weights, const std::string& name) const;
  TensorPtr apply(Ops& ops, const TensorPtr& x, const Conv& c, Act act) const;
  TensorPtr block(Ops& ops, const TensorPtr& x, int index) const;
  // Feature map, and the score map when score_map is not null; both cropped to the image.
  void dense(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels, const StageSink& sink,
             TensorPtr* feature_map, TensorPtr* score_map) const;

  Conv conv1_[4], conv2_[4], down_[4];  // encoder blocks 1..4 (block 1 has no residual branch)
  Conv agg_[4];                         // 1x1 aggregation convolutions
  Conv score_[4];                       // score head
  SddhParams sddh_;
  int dim_ = 0;
};

}  // namespace jkp
