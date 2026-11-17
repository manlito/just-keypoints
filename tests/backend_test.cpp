// Compares a GPU backend with the CPU backend on generated images: every intermediate tensor and the
// final keypoints, scores and descriptors, across the selection modes and pixel formats.
//   backend_test WEIGHTS.jkpw cuda|metal|webgpu
// Needs no golden files. The CPU backend is itself checked against the original by golden_test and
// raco_golden_test. With raco.jkpw it runs the RaCo graph instead of ALIKED.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/aliked/aliked.h"
#include "models/raco/raco.h"
#include "test_backends.h"
#include "test_picture.h"

namespace {

int g_failed = 0;

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0;
  for (size_t i = 0; i < a.size(); ++i) d = std::fmax(d, std::fabs(static_cast<double>(a[i]) - b[i]));
  return d;
}

// Returns the difference, or -1 when the sizes differ.
double compare(const char* name, const std::vector<float>& ref, const std::vector<float>& got, double tol,
               bool verbose) {
  bool same_size = ref.size() == got.size();
  double d = same_size ? max_abs_diff(ref, got) : -1;
  bool ok = same_size && d <= tol;
  if (!same_size) std::printf("    %-12s FAIL (size: cpu %zu, gpu %zu)\n", name, ref.size(), got.size());
  else if (!ok || verbose) std::printf("    %-12s %s  max|diff| %.2e  (tol %.0e)\n", name, ok ? "ok  " : "FAIL", d, tol);
  g_failed += !ok;
  return d;
}

struct Case {
  const char* name;
  int w, h, channels, extra_stride;
  int top_k;
  float scores_th;
  int n_limit;
  bool flat;  // constant image: every score ties
};

struct Run {
  std::map<std::string, std::vector<float>> stages;
  jkp::Features out;
};

Run run(const jkp::Aliked& model, jkp::Ops& ops, const Case& c, const std::vector<uint8_t>& image, int stride) {
  Run r;
  jkp::DetectParams detect;
  detect.top_k = c.top_k, detect.scores_th = c.scores_th, detect.n_limit = c.n_limit;
  model.extract(ops, image.data(), c.w, c.h, stride, c.channels, detect, &r.out,
                [&](const char* name, const jkp::TensorPtr& t) {
                  std::vector<float>& v = r.stages[name];
                  v.resize(static_cast<size_t>(t->c) * t->h * t->w);
                  t->read(v.data());
                });
  return r;
}

// RaCo: values span many orders of magnitude, so stages and outputs are compared relative to the largest
// CPU value.
double compare_rel(const char* name, const std::vector<float>& ref, const std::vector<float>& got, double tol,
                   bool verbose) {
  if (ref.size() != got.size()) {
    std::printf("    %-12s FAIL (size: cpu %zu, gpu %zu)\n", name, ref.size(), got.size());
    ++g_failed;
    return -1;
  }
  double m = 0;
  for (float v : ref) m = std::fmax(m, std::fabs(static_cast<double>(v)));
  const double d = max_abs_diff(ref, got) / std::fmax(m, 1e-30);
  const bool ok = d <= tol;
  if (!ok || verbose) std::printf("    %-12s %s  relative %.2e  (tol %.0e)\n", name, ok ? "ok  " : "FAIL", d, tol);
  g_failed += !ok;
  return d;
}

struct RacoCase {
  const char* name;
  int w, h, channels, extra_stride;
  int top_k;  // <= 0: the default of the weights file
  bool ranker, covariance, flat;
};

int run_raco(const jkp::Weights& weights, const std::string& backend) {
  const RacoCase cases[] = {
      {"rgb, odd size", 203, 157, 3, 0, 0, true, true, false},
      {"multiple of 32", 256, 192, 3, 0, 0, true, true, false},
      {"top-k", 203, 157, 3, 0, 150, true, true, false},
      {"top-k beyond the image", 40, 33, 3, 0, 5000, true, true, false},
      {"gray with row padding", 131, 77, 1, 5, 0, true, true, false},
      {"rgba with row padding", 131, 77, 4, 7, 0, true, true, false},
      {"no ranker", 203, 157, 3, 0, 0, false, true, false},
      {"no covariance", 203, 157, 3, 0, 0, true, false, false},
      {"detector only", 203, 157, 3, 0, 0, false, false, false},
      // Not flat for the network: the border padding breaks the ties (ops_test covers a flat logit map).
      {"uniform gray image", 150, 100, 3, 0, 500, true, true, true},
      {"tiny image", 9, 7, 3, 0, 0, true, true, false},
      {"large, cut at top-k", 1024, 768, 3, 0, 5000, true, true, false},
      {"large, every maximum", 1024, 768, 3, 0, 100000, true, true, false},
  };
  jkp::Raco model(weights);
  auto cpu = jkp::make_cpu_ops(0);
  auto gpu = make_backend(backend);
  for (const RacoCase& c : cases) {
    const int stride = c.w * c.channels + c.extra_stride;
    std::vector<uint8_t> image = picture(c.w, c.h, c.channels, stride, 12345u + static_cast<uint32_t>(c.w));
    if (c.flat) std::fill(image.begin(), image.end(), static_cast<uint8_t>(128));
    jkp::RacoOptions options;
    options.top_k = c.top_k, options.ranker = c.ranker, options.covariance = c.covariance;
    Run r[2];
    jkp::Ops* ops[2] = {cpu.get(), gpu.get()};
    for (int b = 0; b < 2; ++b)
      model.extract(*ops[b], image.data(), c.w, c.h, stride, c.channels, options, &r[b].out,
                    [&](const char* name, const jkp::TensorPtr& t) {
                      std::vector<float>& v = r[b].stages[name];
                      v.resize(static_cast<size_t>(t->c) * t->h * t->w);
                      t->read(v.data());
                    });
    const int before = g_failed;
    std::printf("%s (%dx%d): %d keypoints (cpu), %d (gpu)\n", c.name, c.w, c.h, r[0].out.n, r[1].out.n);
    double worst = 0;
    for (const auto& stage : r[0].stages)
      worst = std::fmax(worst, compare_rel(stage.first.c_str(), stage.second, r[1].stages[stage.first], 1e-5, false));
    // Keypoints are ordered by score, and scores that differ by float32 rounding swap places (or fall on
    // the other side of the top-k cut). So pair each CPU keypoint with the nearest unused GPU keypoint.
    // Without a cut (fewer maxima than top_k) every keypoint must pair; with one, a keypoint may lack a
    // partner only among the last ones of its list, where it ties with the cut.
    const jkp::Features &a = r[0].out, &g = r[1].out;
    const int top_k = c.top_k > 0 ? c.top_k : weights.meta_int("top_k");
    const bool cut = a.n >= top_k;
    const int tail = std::max(2, a.n / 500);  // 0.2 %
    jkp::Features pa, pg;                    // paired, in the CPU's order
    std::vector<char> used(g.n, 0);
    int unpaired = 0, misplaced = 0;
    for (int i = 0; i < a.n; ++i) {
      int best = -1;
      double best_d = 0.01;  // within 0.1 px
      for (int j = 0; j < g.n; ++j) {
        if (used[j]) continue;
        const double dx = g.xy[2 * j] - a.xy[2 * i], dy = g.xy[2 * j + 1] - a.xy[2 * i + 1];
        if (dx * dx + dy * dy < best_d) best_d = dx * dx + dy * dy, best = j;
      }
      if (best < 0) {
        ++unpaired;
        misplaced += !cut || i < a.n - tail;
        continue;
      }
      used[best] = 1;
      for (const auto& pair : {std::make_pair(&a, i), std::make_pair(&g, best)}) {
        const jkp::Features& f = *pair.first;
        jkp::Features& p = pair.first == &a ? pa : pg;
        const int k = pair.second;
        p.xy.insert(p.xy.end(), &f.xy[2 * k], &f.xy[2 * k] + 2);
        p.score.push_back(f.score[k]);
        if (!f.rank.empty()) p.rank.push_back(f.rank[k]);
        if (!f.cov.empty()) p.cov.insert(p.cov.end(), &f.cov[4 * k], &f.cov[4 * k] + 4);
      }
    }
    for (int j = 0; j < g.n; ++j) misplaced += !used[j] && (!cut || j < g.n - tail);
    if (a.n != g.n || misplaced > 0) {
      std::printf("    %-12s FAIL (%d unpaired of %d, %d not at the cut)\n", "selection", unpaired, a.n, misplaced);
      ++g_failed;
    }
    // Scores, ranker and covariances are bilinear samples at the refined position, so position differences
    // of ~1e-4 px move them by more than the maps differ.
    const double dxy = compare("keypoints", pa.xy, pg.xy, 1e-3, false);
    const double dscore = compare_rel("scores", pa.score, pg.score, 1e-4, false);
    const double drank = compare_rel("ranker", pa.rank, pg.rank, 1e-4, false);
    const double dcov = compare_rel("covariances", pa.cov, pg.cov, 1e-4, false);
    std::printf("    %s  %d unpaired; stages %.1e, position %.1e px, score %.1e, ranker %.1e, covariance %.1e"
                " (relative but position)\n",
                g_failed == before ? "ok  " : "FAIL", unpaired, worst, dxy, dscore, drank, dcov);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string backend = argc == 3 ? argv[2] : "";
  if (backend != "cuda" && backend != "metal" && backend != "webgpu") {
    std::fprintf(stderr, "usage: backend_test WEIGHTS.jkpw cuda|metal|webgpu\n");
    return 2;
  }
  const Case cases[] = {
      {"threshold, rgb, odd size", 203, 157, 3, 0, -1, 0.2f, 5000, false},
      {"threshold, multiple of 32", 256, 192, 3, 0, -1, 0.2f, 5000, false},
      {"threshold over the cap", 203, 157, 3, 0, -1, 0.2f, 20, false},
      {"cap of zero", 96, 64, 3, 0, -1, 0.2f, 0, false},
      {"mean-score fallback", 203, 157, 3, 0, -1, 0.9999f, 5000, false},
      {"threshold disabled", 120, 90, 3, 0, -1, 0.0f, 5000, false},
      {"top-k", 203, 157, 3, 0, 150, 0.2f, 5000, false},
      {"top-k beyond the maxima", 96, 64, 3, 0, 3000, 0.2f, 5000, false},
      {"top-k beyond the image", 40, 33, 3, 0, 5000, 0.2f, 5000, false},
      {"gray with row padding", 131, 77, 1, 5, -1, 0.2f, 5000, false},
      {"rgba with row padding", 131, 77, 4, 7, -1, 0.2f, 5000, false},
      {"large, many keypoints", 1024, 768, 3, 0, -1, 0.05f, 20000, false},
      {"large, over the cap", 1024, 768, 3, 0, -1, 0.05f, 500, false},
      {"flat image, threshold", 150, 100, 3, 0, -1, 0.2f, 5, true},
      {"flat image, top-k", 150, 100, 3, 0, 500, 0.2f, 5000, true},
      {"tiny image", 9, 7, 3, 0, -1, 0.2f, 5000, false},
  };
  try {
    jkp::Weights weights = jkp::Weights::from_file(argv[1]);
    if (weights.meta("model") == "raco") {
      run_raco(weights, backend);
      std::printf(g_failed ? "FAILED (%d)\n" : "all cases match\n", g_failed);
      return g_failed ? 1 : 0;
    }
    jkp::Aliked model(weights);
    auto cpu = jkp::make_cpu_ops(0);
    auto gpu = make_backend(backend);
    for (const Case& c : cases) {
      int stride = c.w * c.channels + c.extra_stride;
      std::vector<uint8_t> image = picture(c.w, c.h, c.channels, stride, 12345u + static_cast<uint32_t>(c.w));
      if (c.flat) std::fill(image.begin(), image.end(), static_cast<uint8_t>(128));
      Run a = run(model, *cpu, c, image, stride), b = run(model, *gpu, c, image, stride);
      int before = g_failed;
      std::printf("%s (%dx%d): %d keypoints (cpu), %d (gpu)\n", c.name, c.w, c.h, a.out.n, b.out.n);
      double worst = 0;
      for (const auto& stage : a.stages)
        worst = std::fmax(worst, compare(stage.first.c_str(), stage.second, b.stages[stage.first], 1e-4, false));
      // Same keypoints in the same order.
      double dxy = compare("keypoints", a.out.xy, b.out.xy, 1e-2, false);
      double dscore = compare("scores", a.out.score, b.out.score, 1e-4, false);
      double ddesc = compare("descriptors", a.out.desc, b.out.desc, 1e-4, false);
      std::printf("    %s  max|diff|: stages %.1e, position %.1e px, score %.1e, descriptor %.1e\n",
                  g_failed == before ? "ok  " : "FAIL", worst, dxy, dscore, ddesc);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all cases match\n", g_failed);
  return g_failed ? 1 : 0;
}
