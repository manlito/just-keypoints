// Compares every stage of a backend against golden tensors dumped from the original ALIKED.
//   golden_test WEIGHTS.jkpw GOLDEN.jkpt [cpu|cuda|metal|webgpu]
// Generate golden files with: python tools/dump_golden.py <images>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/aliked/aliked.h"
#include "test_backends.h"

namespace {

int g_failed = 0;

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
  bool ok = max_diff <= tol;
  std::printf("%-14s %s  max|diff| %.2e  (max|ref| %.2e, tol %.0e)\n", name, ok ? "ok  " : "FAIL", max_diff, max_ref,
              tol);
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string backend = argc == 4 ? argv[3] : "cpu";
  if (argc < 3 || argc > 4 || (backend != "cpu" && backend != "cuda" && backend != "metal" && backend != "webgpu")) {
    std::fprintf(stderr, "usage: golden_test WEIGHTS.jkpw GOLDEN.jkpt [cpu|cuda|metal|webgpu]\n");
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

    jkp::Aliked model(weights);
    auto ops = make_backend(backend);
    jkp::DetectParams detect;
    detect.top_k = golden.meta_int("top_k");
    detect.scores_th = std::stof(golden.meta("scores_th"));
    detect.n_limit = golden.meta_int("n_limit");

    jkp::Features out;
    model.extract(*ops, image.data(), w, h, 0, 3, detect, &out, [&](const char* name, const jkp::TensorPtr& t) {
      std::vector<float> v(static_cast<size_t>(t->c) * t->h * t->w);
      t->read(v.data());
      check(name, v.data(), v.size(), golden.find(name), 1e-4);
    });
    check("keypoints", out.xy.data(), out.xy.size(), golden.find("keypoints"), 1e-2);
    check("scores", out.score.data(), out.score.size(), golden.find("scores"), 1e-4);
    check("descriptors", out.desc.data(), out.desc.size(), golden.find("descriptors"), 1e-4);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all stages match\n", g_failed);
  return g_failed ? 1 : 0;
}
