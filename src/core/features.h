// Output of a model graph for one image.
#pragma once

#include <vector>

namespace jkp {

struct Features {
  int n = 0, dim = 0;
  std::vector<float> xy;     // [n][2] pixels, (0, 0) = centre of the top-left pixel
  std::vector<float> score;  // [n]
  std::vector<float> desc;   // [n][dim]; empty for a detector without descriptors (dim = 0)
  std::vector<float> rank;   // [n] ranker score, or empty (RaCo)
  std::vector<float> cov;    // [n][2][2] keypoint covariance in pixels^2, or empty (RaCo)
};

}  // namespace jkp
