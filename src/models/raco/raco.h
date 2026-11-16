// RaCo graph (detector with ranker and covariance heads, no descriptors), written against the
// backend-independent Ops interface. Mirrors tools/raco_folded.py, which is the executable specification.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/features.h"
#include "core/ops.h"
#include "core/weights.h"

namespace jkp {

struct RacoOptions {
  int top_k = 0;           // number of keypoints; <= 0 uses the default stored in the weights file (2048)
  bool ranker = true;      // run the ranker head (Features::rank)
  bool covariance = true;  // run the covariance head (Features::cov)
};

class Raco {
 public:
  explicit Raco(const Weights& weights);

  // Called with every intermediate tensor when set (golden tests).
  using StageSink = std::function<void(const char* name, const TensorPtr& tensor)>;

  // data: 8-bit image with 1, 3 or 4 channels.
  void extract(Ops& ops, const uint8_t* data, int width, int height, int stride, int channels,
               const RacoOptions& options, Features* out, const StageSink& sink = nullptr) const;

 private:
  struct Conv {
    const Param* w = nullptr;
    const Param* b = nullptr;  // may be null
  };
  struct Block {
    Conv conv1, conv2, down;  // down.w is null for a block without a residual branch
  };
  Conv conv(const Weights& weights, const std::string& name) const;
  Block block(const Weights& weights, const std::string& name) const;
  TensorPtr apply(Ops& ops, const TensorPtr& x, const Block& b) const;

  const Param *mean_, *std_;
  Block block_[4];           // encoder
  Conv agg_[4];              // aggregation convolutions (1x1, then 3x3)
  Conv score_[4];            // score head
  std::vector<Block> ranker_;
  Conv ranker_out_;          // 5x5, reflect padding
  Conv cov_[4];              // three 3x3 with reflect padding, then 1x1
  RacoDetectParams detect_;
};

}  // namespace jkp
