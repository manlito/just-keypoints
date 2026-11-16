// jkp command line tool.
//   jkp extract IMAGE [--out OUT.jkpt] [options]   extract keypoints and descriptors (raco: ranker, covariances)
//   jkp bench   IMAGE [--runs N] [--warmup N] [options]   time the extraction (warm-up runs are not counted)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/weights.h"
#include "jkp/jkp.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: jkp extract IMAGE [--out OUT.jkpt] [options]\n"
               "       jkp bench   IMAGE [--runs N] [--warmup N] [options]   (default 10 runs after 2 warm-up runs)\n"
               "options:\n"
               "  --image PATH      input image (same as the positional argument)\n"
               "  --backend NAME    cpu (default), cuda or metal\n"
               "  --device N        GPU index for --backend cuda or metal (default 0)\n"
               "  --precision P     fp32 (default, matches the original) or fp16 (cuda and metal: faster)\n"
               "  --weights PATH    .jkpw file (default: weights embedded in the build)\n"
               "  --model NAME      aliked-n16 (default) or raco; picks the embedded weights\n"
               "  --descriptor NAME aliked-n16: describe raco keypoints with it\n"
               "  --descriptor-weights PATH   .jkpw file for --descriptor (default: embedded)\n"
               "  --no-ranker, --no-covariance   raco: skip that head\n"
               "  --top-k K         keep the K best keypoints (default: threshold mode; raco: 2048)\n"
               "  --scores-th T     score threshold (default 0.2)\n"
               "  --n-limit N       maximum keypoints in threshold mode (default 5000)\n"
               "  --threads N       CPU worker threads (default: all)\n"
               "  --gray            feed the image as 8-bit grayscale\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  std::string cmd = argv[1], image_path, out_path, weights_path, model, descriptor, descriptor_weights;
  int runs = 10, warmup = 2;
  bool gray = false;
  jkp_config config;
  jkp_config_default(&config);
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--image") image_path = value();
    else if (a == "--out") out_path = value();
    else if (a == "--weights") weights_path = value();
    else if (a == "--model") model = value();
    else if (a == "--descriptor") descriptor = value();
    else if (a == "--descriptor-weights") descriptor_weights = value();
    else if (a == "--no-ranker") config.skip_ranker = 1;
    else if (a == "--no-covariance") config.skip_covariance = 1;
    else if (a == "--backend") {
      std::string name = value();
      if (name == "cpu") config.backend = JKP_BACKEND_CPU;
      else if (name == "cuda") config.backend = JKP_BACKEND_CUDA;
      else if (name == "metal") config.backend = JKP_BACKEND_METAL;
      else return usage();
    }
    else if (a == "--device") config.device = std::atoi(value());
    else if (a == "--precision") {
      std::string name = value();
      if (name == "fp32") config.precision = JKP_PRECISION_FP32;
      else if (name == "fp16") config.precision = JKP_PRECISION_FP16;
      else return usage();
    }
    else if (a == "--top-k") config.top_k = std::atoi(value());
    else if (a == "--scores-th") config.score_threshold = static_cast<float>(std::atof(value()));
    else if (a == "--n-limit") config.max_keypoints = std::atoi(value());
    else if (a == "--threads") config.num_threads = std::atoi(value());
    else if (a == "--runs") runs = std::atoi(value());
    else if (a == "--warmup") warmup = std::atoi(value());
    else if (a == "--gray") gray = true;
    else if (a.rfind("--", 0) == 0) return usage();
    else image_path = a;
  }
  if ((cmd != "extract" && cmd != "bench") || image_path.empty()) return usage();
  if (!weights_path.empty()) config.weights_path = weights_path.c_str();
  if (!model.empty()) config.model = model.c_str();
  if (!descriptor.empty()) config.descriptor = descriptor.c_str();
  if (!descriptor_weights.empty()) config.descriptor_weights_path = descriptor_weights.c_str();

  int w = 0, h = 0, n = 0;
  unsigned char* pixels = stbi_load(image_path.c_str(), &w, &h, &n, gray ? 1 : 3);
  if (!pixels) {
    std::fprintf(stderr, "cannot read image '%s': %s\n", image_path.c_str(), stbi_failure_reason());
    return 1;
  }
  jkp_image image{pixels, w, h, 0, gray ? JKP_FORMAT_GRAY8 : JKP_FORMAT_RGB8};

  jkp_ctx* ctx = jkp_create(&config);
  if (!ctx) {
    std::fprintf(stderr, "jkp_create: %s\n", jkp_last_error());
    return 1;
  }
  jkp_result r{};
  std::vector<double> ms;
  // bench: the first runs compile kernels and fill the buffer pool; they are not counted.
  const int skipped = cmd == "bench" ? std::max(0, warmup) : 0;
  int iterations = cmd == "bench" ? std::max(1, runs) : 1;
  for (int i = -skipped; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    if (jkp_extract(ctx, &image, &r) != 0) {
      std::fprintf(stderr, "jkp_extract: %s\n", jkp_last_error());
      return 1;
    }
    if (i >= 0) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }

  if (cmd == "bench") {
    std::sort(ms.begin(), ms.end());
    double sum = 0;
    for (double v : ms) sum += v;
    std::printf("%dx%d, %d keypoints, %d runs: min %.1f ms, median %.1f ms, mean %.1f ms\n", w, h, r.n, iterations,
                ms.front(), ms[ms.size() / 2], sum / ms.size());
  } else {
    std::printf("%dx%d: %d keypoints, descriptor dim %d, %.1f ms\n", w, h, r.n, r.dim, ms[0]);
    if (!out_path.empty()) {
      try {
        std::vector<jkp::NamedTensor> tensors = {{"keypoints", {r.n, 2}, r.xy}, {"scores", {r.n}, r.score}};
        if (r.dim > 0) tensors.push_back({"descriptors", {r.n, r.dim}, r.desc});
        if (r.rank) tensors.push_back({"ranker", {r.n}, r.rank});
        if (r.cov) tensors.push_back({"covariances", {r.n, 2, 2}, r.cov});
        jkp::write_container(out_path,
                             {{"kind", "features"}, {"width", std::to_string(w)}, {"height", std::to_string(h)}},
                             tensors);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
      }
    }
  }
  jkp_destroy(ctx);
  stbi_image_free(pixels);
  return 0;
}
