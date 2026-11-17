// FP16 against FP32 on the same GPU backend: how many keypoints survive, how far they move and how much
// the descriptors change. Fails when the loss exceeds the bounds below.
//   precision_test WEIGHTS.jkpw cuda|metal|webgpu [GOLDEN.jkpt ...]
// Runs on generated images, plus the image stored in each golden file given (real photographs).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/aliked/aliked.h"
#include "test_backends.h"
#include "test_picture.h"

namespace {

// Bounds. Measured values are far inside them (see PLAN.md, M5); they catch a broken FP16 path, e.g.
// half arithmetic where float was meant, not small drifts.
constexpr double kMinRepeat = 0.97;       // keypoints of one run found within 1 px in the other, both ways
constexpr double kMaxMedianShift = 0.02;  // px, median over the keypoints found in both
constexpr double kMinMeanCosine = 0.999;  // descriptors of the keypoints found in both
constexpr double kMinCosine = 0.95;       // worst pair; those are neighbouring maxima almost 1 px apart

struct Stats {
  int n32 = 0, n16 = 0, common = 0;
  double repeat = 1, median_shift = 0, max_shift = 0, max_score = 0, mean_cos = 1, min_cos = 1;
};

// For each keypoint of a, the nearest keypoint of b (index, squared distance); -1 when b is empty.
std::vector<std::pair<int, double>> nearest(const jkp::Features& a, const jkp::Features& b) {
  std::vector<std::pair<int, double>> out(a.n, {-1, 1e30});
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < b.n; ++j) {
      double dx = a.xy[2 * i] - b.xy[2 * j], dy = a.xy[2 * i + 1] - b.xy[2 * j + 1], d = dx * dx + dy * dy;
      if (d < out[i].second) out[i] = {j, d};
    }
  return out;
}

Stats compare(const jkp::Features& f32, const jkp::Features& f16) {
  Stats s;
  s.n32 = f32.n, s.n16 = f16.n;
  auto ab = nearest(f32, f16), ba = nearest(f16, f32);
  int found_ab = 0, found_ba = 0;
  for (const auto& m : ab) found_ab += m.first >= 0 && m.second <= 1.0;
  for (const auto& m : ba) found_ba += m.first >= 0 && m.second <= 1.0;
  if (f32.n + f16.n > 0) s.repeat = static_cast<double>(found_ab + found_ba) / (f32.n + f16.n);
  std::vector<double> shifts;
  double cos_sum = 0;
  for (int i = 0; i < f32.n; ++i) {
    const int j = ab[i].first;
    if (j < 0 || ab[i].second > 1.0 || ba[j].first != i) continue;  // mutual nearest within 1 px
    shifts.push_back(std::sqrt(ab[i].second));
    s.max_score = std::fmax(s.max_score, std::fabs(static_cast<double>(f32.score[i]) - f16.score[j]));
    double dot = 0;
    for (int d = 0; d < f32.dim; ++d)
      dot += static_cast<double>(f32.desc[static_cast<size_t>(i) * f32.dim + d]) * f16.desc[static_cast<size_t>(j) * f16.dim + d];
    cos_sum += dot;
    s.min_cos = std::fmin(s.min_cos, dot);
  }
  s.common = static_cast<int>(shifts.size());
  if (s.common) {
    std::sort(shifts.begin(), shifts.end());
    s.median_shift = shifts[shifts.size() / 2];
    s.max_shift = shifts.back();
    s.mean_cos = cos_sum / s.common;
  }
  return s;
}

struct Image {
  std::string name;
  int w, h;
  std::vector<uint8_t> rgb;
};

}  // namespace

int main(int argc, char** argv) {
  const std::string backend = argc >= 3 ? argv[2] : "";
  if (backend != "cuda" && backend != "metal" && backend != "webgpu") {
    std::fprintf(stderr, "usage: precision_test WEIGHTS.jkpw cuda|metal|webgpu [GOLDEN.jkpt ...]\n");
    return 2;
  }
  int failed = 0;
  try {
    jkp::Weights weights = jkp::Weights::from_file(argv[1]);
    jkp::Aliked model(weights);
    auto fp32 = make_backend(backend), fp16 = make_backend(backend + ":fp16");

    std::vector<Image> images;
    const int sizes[][2] = {{320, 240}, {640, 480}, {517, 389}, {1024, 768}};
    for (const auto& wh : sizes)
      images.push_back({"generated", wh[0], wh[1], picture(wh[0], wh[1], 3, wh[0] * 3, 777u + static_cast<uint32_t>(wh[0]))});
    for (int i = 3; i < argc; ++i) {
      jkp::Weights golden = jkp::Weights::from_file(argv[i]);
      const jkp::Param& f = golden.get("image_f32");
      Image im{argv[i], golden.meta_int("width"), golden.meta_int("height"), std::vector<uint8_t>(f.size())};
      for (size_t k = 0; k < im.rgb.size(); ++k) im.rgb[k] = static_cast<uint8_t>(f.data[k]);
      const size_t slash = im.name.find_last_of('/');
      if (slash != std::string::npos) im.name = im.name.substr(slash + 1);
      images.push_back(std::move(im));
    }

    std::printf("%-26s %-9s %-10s %6s %6s %7s %9s %9s %9s %10s %10s\n", "image", "size", "mode", "fp32", "fp16",
                "repeat", "med px", "max px", "score", "mean 1-cos", "max 1-cos");
    for (const Image& im : images) {
      for (int mode = 0; mode < 2; ++mode) {
        jkp::DetectParams detect;  // threshold mode by default
        if (mode == 1) detect.top_k = 1000;
        jkp::Features a, b;
        model.extract(*fp32, im.rgb.data(), im.w, im.h, 0, 3, detect, &a);
        model.extract(*fp16, im.rgb.data(), im.w, im.h, 0, 3, detect, &b);
        Stats s = compare(a, b);
        const bool ok = s.repeat >= kMinRepeat && s.median_shift <= kMaxMedianShift && s.mean_cos >= kMinMeanCosine &&
                        s.min_cos >= kMinCosine && s.common > 0;
        failed += !ok;
        char size[32];
        std::snprintf(size, sizeof(size), "%dx%d", im.w, im.h);
        std::printf("%-26s %-9s %-10s %6d %6d %6.2f%% %9.4f %9.4f %9.1e %10.1e %10.1e%s\n", im.name.c_str(), size,
                    mode ? "top-k 1000" : "threshold", s.n32, s.n16, 100 * s.repeat, s.median_shift, s.max_shift,
                    s.max_score, 1 - s.mean_cos, 1 - s.min_cos, ok ? "" : "  FAIL");
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(failed ? "FAILED (%d)\n" : "fp16 within bounds\n", failed);
  return failed ? 1 : 0;
}
