// Compares every stage of the XFeat graph against golden tensors dumped from the original XFeat.
//   xfeat_golden_test WEIGHTS.jkpw GOLDEN.jkpt [cpu|metal|metal:fp16|...]
// Generate golden files with: python tools/dump_golden.py --model xfeat <images>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/xfeat/xfeat.h"
#include "test_backends.h"

namespace {

int g_failed = 0;

// Maximum error relative to the largest golden value.
void check(const char* name, const float* got, size_t got_n, const jkp::Param* ref, double tol) {
  if (!ref || ref->size() != got_n) {
    std::printf("%-12s FAIL (size: got %zu, golden %zu)\n", name, got_n, ref ? ref->size() : 0);
    ++g_failed;
    return;
  }
  double max_diff = 0, max_ref = 0;
  for (size_t i = 0; i < got_n; ++i) {
    max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(got[i]) - ref->data[i]));
    max_ref = std::fmax(max_ref, std::fabs(static_cast<double>(ref->data[i])));
  }
  double rel = max_diff / std::fmax(max_ref, 1e-30);
  bool ok = rel <= tol;
  std::printf("%-12s %s  max|diff| %.2e  relative %.2e  (max|ref| %.2e, tol %.0e)\n", name, ok ? "ok  " : "FAIL",
              max_diff, rel, max_ref, tol);
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: xfeat_golden_test WEIGHTS.jkpw GOLDEN.jkpt [backend]\n");
    return 2;
  }
  try {
    jkp::Weights weights = jkp::Weights::from_file(argv[1]);
    jkp::Weights golden = jkp::Weights::from_file(argv[2]);
    const int w = golden.meta_int("width"), h = golden.meta_int("height");
    const std::string backend = argc == 4 ? argv[3] : "cpu";
    const bool fp16 = backend.find("fp16") != std::string::npos;

    const jkp::Param& image_f = golden.get("image_f32");
    std::vector<uint8_t> image(image_f.size());
    for (size_t i = 0; i < image.size(); ++i) image[i] = static_cast<uint8_t>(image_f.data[i]);

    jkp::Xfeat model(weights);
    auto ops = make_backend(backend);
    jkp::Features out;
    model.extract(*ops, image.data(), w, h, 0, 3, golden.meta_int("top_k"), &out,
                  [&](const char* name, const jkp::TensorPtr& t) {
                    std::vector<float> v(static_cast<size_t>(t->c) * t->h * t->w);
                    t->read(v.data());
                    check(name, v.data(), v.size(), golden.find(name), fp16 ? 3e-2 : 1e-4);
                  });

    // Keypoints are pixel positions of the resized image, so the same keypoint is at the same position.
    // Scores that differ by float32 rounding can swap places or fall on the other side of the top-k cut.
    const jkp::Param &ref_xy = golden.get("keypoints"), &ref_score = golden.get("scores"),
                     &ref_desc = golden.get("descriptors");
    const int n_ref = ref_xy.shape[0], dim = ref_desc.shape[1];
    std::map<std::pair<float, float>, int> ours;
    for (int j = 0; j < out.n; ++j) ours[{out.xy[2 * j], out.xy[2 * j + 1]}] = j;
    int unpaired = 0;
    double max_pos = 0, max_score = 0, min_cos = 1;
    for (int i = 0; i < n_ref; ++i) {
      int best = -1;
      double best_d = 1e-4;
      auto it = ours.find({ref_xy.data[2 * i], ref_xy.data[2 * i + 1]});
      if (it != ours.end()) best = it->second, best_d = 0;
      for (int j = 0; best < 0 && j < out.n; ++j) {  // not bit-identical: nearest within 0.01 px
        double dx = out.xy[2 * j] - ref_xy.data[2 * i], dy = out.xy[2 * j + 1] - ref_xy.data[2 * i + 1];
        if (dx * dx + dy * dy < best_d) best = j;
      }
      if (best < 0) {
        ++unpaired;
        continue;
      }
      max_pos = std::fmax(max_pos, std::hypot(out.xy[2 * best] - ref_xy.data[2 * i],
                                              out.xy[2 * best + 1] - ref_xy.data[2 * i + 1]));
      max_score = std::fmax(max_score, std::fabs(out.score[best] - ref_score.data[i]));
      double dot = 0;
      for (int c = 0; c < dim; ++c) dot += static_cast<double>(out.desc[best * dim + c]) * ref_desc.data[i * dim + c];
      min_cos = std::fmin(min_cos, dot);
    }
    // At most 0.2 % of the keypoints at the top-k cut (FP16: 2 %).
    const bool count_ok = std::abs(out.n - n_ref) * (fp16 ? 50 : 500) <= n_ref && unpaired * (fp16 ? 50 : 500) <= n_ref;
    std::printf("%-12s %s  %d keypoints, golden %d, %d unpaired\n", "selection", count_ok ? "ok  " : "FAIL", out.n,
                n_ref, unpaired);
    const bool kp_ok = max_pos <= 1e-4 && max_score <= (fp16 ? 5e-2 : 1e-4) && min_cos >= (fp16 ? 0.99 : 1 - 1e-6);
    std::printf("%-12s %s  position %.2e px, score %.2e, 1 - cosine %.2e\n", "keypoints", kp_ok ? "ok  " : "FAIL",
                max_pos, max_score, 1 - min_cos);
    g_failed += !count_ok + !kp_ok;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all stages match\n", g_failed);
  return g_failed ? 1 : 0;
}
