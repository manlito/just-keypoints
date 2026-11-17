// Compares every stage of the RaCo graph against golden tensors dumped from the original RaCo.
//   raco_golden_test WEIGHTS.jkpw GOLDEN.jkpt [cpu|metal|metal:fp16|...]
// Generate golden files with: python tools/dump_golden.py --model raco <images>
// Backends with the RaCo operations: cpu, cuda, metal, webgpu.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/raco/raco.h"
#include "test_backends.h"

namespace {

int g_failed = 0;

// Maximum error relative to the largest golden value (RaCo tensors span 1e-9 .. 1e3).
void check(const char* name, const float* got, size_t got_n, const jkp::Param* ref, double tol) {
  if (!ref || ref->size() != got_n) {
    std::printf("%-14s FAIL (size: got %zu, golden %zu)\n", name, got_n, ref ? ref->size() : 0);
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
  std::printf("%-14s %s  max|diff| %.2e  relative %.2e  (max|ref| %.2e, tol %.0e)\n", name, ok ? "ok  " : "FAIL",
              max_diff, rel, max_ref, tol);
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: raco_golden_test WEIGHTS.jkpw GOLDEN.jkpt [backend]\n");
    return 2;
  }
  try {
    jkp::Weights weights = jkp::Weights::from_file(argv[1]);
    jkp::Weights golden = jkp::Weights::from_file(argv[2]);
    int w = golden.meta_int("width"), h = golden.meta_int("height");

    // The C++ reader only loads float32 tensors, so the dump carries the 8-bit image as "image_f32" (HxWx3).
    const jkp::Param& image_f = golden.get("image_f32");
    std::vector<uint8_t> image(image_f.size());
    for (size_t i = 0; i < image.size(); ++i) image[i] = static_cast<uint8_t>(image_f.data[i]);

    jkp::Raco model(weights);
    auto ops = make_backend(argc == 4 ? argv[3] : "cpu");
    jkp::RacoOptions options;
    options.top_k = golden.meta_int("top_k");

    jkp::Features out;
    model.extract(*ops, image.data(), w, h, 0, 3, options, &out, [&](const char* name, const jkp::TensorPtr& t) {
      const size_t plane = static_cast<size_t>(t->h) * t->w;
      std::vector<float> v(t->c * plane);
      t->read(v.data());
      if (std::string(name) == "cholesky_raw") {
        // The golden map has softplus applied to the diagonal elements (channels 0 and 2).
        for (int c = 0; c < 3; c += 2)
          for (size_t i = 0; i < plane; ++i) {
            float& x = v[c * plane + i];
            x = x > 20.0f ? x : std::log1p(std::exp(x));
          }
        name = "cholesky_map";
      }
      check(name, v.data(), v.size(), golden.find(name), 1e-4);
    });
    // Keypoints are ordered by score, and scores that differ by float32 rounding swap places (or fall on
    // the other side of the top-k cut). So pair each golden keypoint with our nearest one, then compare.
    const jkp::Param& ref_xy = golden.get("keypoints");
    const int n_ref = ref_xy.shape[0];
    std::vector<float> xy, score, rank, cov, ref_score, ref_rank, ref_cov, ref_kept;
    int unpaired = 0;
    for (int i = 0; i < n_ref; ++i) {
      int best = -1;
      double best_d = 0.25;  // within 0.5 px
      for (int j = 0; j < out.n; ++j) {
        double dx = out.xy[2 * j] - ref_xy.data[2 * i], dy = out.xy[2 * j + 1] - ref_xy.data[2 * i + 1];
        if (dx * dx + dy * dy < best_d) best_d = dx * dx + dy * dy, best = j;
      }
      if (best < 0) {
        ++unpaired;
        continue;
      }
      xy.insert(xy.end(), &out.xy[2 * best], &out.xy[2 * best] + 2);
      score.push_back(out.score[best]);
      rank.push_back(out.rank[best]);
      cov.insert(cov.end(), &out.cov[4 * best], &out.cov[4 * best] + 4);
      ref_kept.insert(ref_kept.end(), ref_xy.data + 2 * i, ref_xy.data + 2 * i + 2);
      ref_score.push_back(golden.get("scores").data[i]);
      ref_rank.push_back(golden.get("ranker").data[i]);
      ref_cov.insert(ref_cov.end(), golden.get("covariances").data + 4 * i, golden.get("covariances").data + 4 * i + 4);
    }
    bool count_ok = out.n == n_ref && unpaired * 500 <= n_ref;  // at most 0.2 % at the cut
    std::printf("%-14s %s  %d keypoints, golden %d, %d unpaired\n", "selection", count_ok ? "ok  " : "FAIL", out.n,
                n_ref, unpaired);
    g_failed += !count_ok;
    auto param = [](const std::vector<float>& v) { return jkp::Param{{static_cast<int>(v.size())}, v.data()}; };
    jkp::Param p_xy = param(ref_kept), p_score = param(ref_score), p_rank = param(ref_rank), p_cov = param(ref_cov);
    check("keypoints", xy.data(), xy.size(), &p_xy, 1e-4);
    check("scores", score.data(), score.size(), &p_score, 1e-3);
    check("ranker", rank.data(), rank.size(), &p_rank, 1e-4);
    check("covariances", cov.data(), cov.size(), &p_cov, 1e-3);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all stages match\n", g_failed);
  return g_failed ? 1 : 0;
}
