// Compares a GPU backend with the CPU backend on generated images: every intermediate tensor and the
// final keypoints, scores and descriptors, across the selection modes and pixel formats.
//   backend_test WEIGHTS.jkpw cuda
// Needs no golden files. The CPU backend is itself checked against the original ALIKED by golden_test.
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

// Deterministic test picture: gradient background, rectangles, discs and a little noise.
std::vector<uint8_t> picture(int w, int h, int channels, int stride, uint32_t seed) {
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
  };
  std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = 60.0f + 80.0f * (c == 0 ? x : y) / (c == 2 ? h : w);
  for (int shape = 0; shape < 40; ++shape) {
    int cx = next() % w, cy = next() % h, rx = 3 + next() % (w / 5 + 1), ry = 3 + next() % (h / 5 + 1);
    bool disc = next() % 3 == 0;
    float color[3] = {static_cast<float>(next() % 256), static_cast<float>(next() % 256), static_cast<float>(next() % 256)};
    for (int y = std::max(0, cy - ry); y < std::min(h, cy + ry); ++y)
      for (int x = std::max(0, cx - rx); x < std::min(w, cx + rx); ++x) {
        float dx = static_cast<float>(x - cx) / rx, dy = static_cast<float>(y - cy) / ry;
        if (disc && dx * dx + dy * dy > 1.0f) continue;
        for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = color[c];
      }
  }
  std::vector<uint8_t> out(static_cast<size_t>(stride) * h, 0xAB);  // row padding holds junk on purpose
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float* p = &rgb[(static_cast<size_t>(y) * w + x) * 3];
      float noise = static_cast<float>(next() % 9) - 4.0f;
      uint8_t* o = &out[static_cast<size_t>(y) * stride + static_cast<size_t>(x) * channels];
      auto u8 = [&](float v) { return static_cast<uint8_t>(std::fmin(255.0f, std::fmax(0.0f, v + noise))); };
      if (channels == 1) o[0] = u8((p[0] + p[1] + p[2]) / 3.0f);
      else o[0] = u8(p[0]), o[1] = u8(p[1]), o[2] = u8(p[2]);
      if (channels == 4) o[3] = static_cast<uint8_t>(next() % 256);  // alpha must be ignored
    }
  return out;
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[2]) != "cuda") {
    std::fprintf(stderr, "usage: backend_test WEIGHTS.jkpw cuda\n");
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
    jkp::Aliked model(weights);
    auto cpu = jkp::make_cpu_ops(0);
    auto gpu = jkp::make_cuda_ops(0);
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
