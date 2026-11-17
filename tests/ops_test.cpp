// Compares each operation of a GPU backend with the CPU backend on its own, on generated tensors and
// random weights, so that a wrong kernel is named directly (backend_test sees only whole stages).
//   ops_test cuda|metal|webgpu[:fp16]
// FP32 tolerance 1e-4 (absolute, values of order 1); FP16 1e-1.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "test_backends.h"
#include "test_picture.h"

namespace {

int g_failed = 0;
double g_tol = 1e-4;

// Random weights that stay alive for the whole run (Param points into them).
struct RandomParams {
  std::vector<std::vector<float>> storage;
  std::vector<jkp::Param> params;
  uint32_t seed = 7;
  float next() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
  }
  const jkp::Param& make(std::vector<int> shape, float scale) {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    storage.emplace_back(n);
    for (float& v : storage.back()) v = next() * scale;
    params.reserve(256);  // references stay valid
    params.push_back({shape, storage.back().data()});
    return params.back();
  }
};

std::vector<float> values(const jkp::TensorPtr& t) {
  std::vector<float> v(static_cast<size_t>(t->c) * t->h * t->w);
  t->read(v.data());
  return v;
}

void check(const std::string& name, const jkp::TensorPtr& a, const jkp::TensorPtr& b) {
  if (a->c != b->c || a->h != b->h || a->w != b->w) {
    std::printf("%-44s FAIL (shape: cpu %dx%dx%d, gpu %dx%dx%d)\n", name.c_str(), a->c, a->h, a->w, b->c, b->h, b->w);
    ++g_failed;
    return;
  }
  const std::vector<float> va = values(a), vb = values(b);
  double d = 0;
  size_t at = 0;
  for (size_t i = 0; i < va.size(); ++i) {
    const double e = std::fabs(static_cast<double>(va[i]) - vb[i]);
    if (e > d || std::isnan(e)) d = e, at = i;
  }
  const bool ok = d <= g_tol;
  std::printf("%-44s %s  max|diff| %.2e", name.c_str(), ok ? "ok  " : "FAIL", d);
  if (!ok)
    std::printf("  at c%zu y%zu x%zu: cpu %g gpu %g", at / (a->h * a->w), at / a->w % a->h, at % a->w, va[at], vb[at]);
  std::printf("\n");
  g_failed += !ok;
}

void check_vec(const std::string& name, const std::vector<float>& a, const std::vector<float>& b) {
  double d = a.size() == b.size() ? 0 : INFINITY;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::fabs(static_cast<double>(a[i]) - b[i]));
  const bool ok = d <= g_tol;
  std::printf("%-44s %s  max|diff| %.2e  (%zu / %zu values)\n", name.c_str(), ok ? "ok  " : "FAIL", d, a.size(), b.size());
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: ops_test cuda|metal|webgpu[:fp16]\n");
    return 2;
  }
  const std::string spec = argv[1];
  if (spec.find(":fp16") != std::string::npos) g_tol = 1e-1;
  try {
    auto cpu = jkp::make_cpu_ops(0);
    auto gpu = make_backend(spec);
    RandomParams rp;
    jkp::Ops* ops[2] = {cpu.get(), gpu.get()};
    // Runs f on both backends and compares the results.
    auto both = [&](const std::string& name, const std::function<jkp::TensorPtr(jkp::Ops&, int)>& f) {
      jkp::TensorPtr r[2];
      for (int b = 0; b < 2; ++b) r[b] = f(*ops[b], b);
      check(name, r[0], r[1]);
      return std::vector<jkp::TensorPtr>{r[0], r[1]};
    };

    const int W = 67, H = 45;
    for (int channels : {1, 3, 4}) {
      const int stride = W * channels + 3;
      const std::vector<uint8_t> px = picture(W, H, channels, stride, 99u + channels);
      both("image " + std::to_string(channels) + " channels",
           [&](jkp::Ops& o, int) { return o.image(px.data(), W, H, stride, channels); });
    }
    const std::vector<uint8_t> px = picture(W, H, 3, W * 3, 5u);
    std::vector<jkp::TensorPtr> img = both("image", [&](jkp::Ops& o, int) { return o.image(px.data(), W, H, 0, 3); });
    std::vector<jkp::TensorPtr> padded = both("pad_replicate", [&](jkp::Ops& o, int b) {
      return o.pad_replicate(img[b], 2, 3, 1, 2);
    });  // 72 x 48, a multiple of 8 for the pooling below
    both("crop", [&](jkp::Ops& o, int b) { return o.crop(padded[b], 3, 2, 31, 17); });

    // Convolutions: every tile size the backends pick, both kernel sizes, with and without bias.
    std::vector<jkp::TensorPtr> x = padded;
    const int couts[] = {3, 8, 16, 24};
    for (int k : {1, 3})
      for (int cout : couts)
        for (bool with_bias : {true, false}) {
          const jkp::Param& w = rp.make({cout, 3, k, k}, 0.5f);
          const jkp::Param& bias = rp.make({cout}, 0.2f);
          for (jkp::Act act : {jkp::Act::Selu, jkp::Act::None}) {
            const std::string name = "conv" + std::to_string(k) + "x" + std::to_string(k) + " 3->" + std::to_string(cout) +
                                     (with_bias ? " bias" : "") + (act == jkp::Act::Selu ? " selu" : "");
            both(name, [&](jkp::Ops& o, int b) { return o.conv2d(x[b], w, with_bias ? &bias : nullptr, act); });
          }
        }
    const jkp::Param& w16 = rp.make({16, 3, 3, 3}, 0.5f);
    const jkp::Param& b16 = rp.make({16}, 0.2f);
    std::vector<jkp::TensorPtr> f16 = both("conv3x3 3->16 (features)", [&](jkp::Ops& o, int b) {
      return o.conv2d(x[b], w16, &b16, jkp::Act::Selu);
    });
    const jkp::Param& ws = rp.make({1, 16, 1, 1}, 0.5f);
    const jkp::Param& bs = rp.make({1}, 0.2f);
    both("conv1x1 16->1 sigmoid", [&](jkp::Ops& o, int b) { return o.conv2d(f16[b], ws, &bs, jkp::Act::Sigmoid); });

    const jkp::Param& ow = rp.make({18, 16, 3, 3}, 0.3f);
    const jkp::Param& ob = rp.make({18}, 0.5f);
    const jkp::Param& dw = rp.make({16, 16, 3, 3}, 0.3f);
    const jkp::Param& db = rp.make({16}, 0.2f);
    std::vector<jkp::TensorPtr> dc = both("deform_conv2d 16->16 selu", [&](jkp::Ops& o, int b) {
      return o.deform_conv2d(f16[b], ow, ob, dw, db, jkp::Act::Selu);
    });
    both("add selu", [&](jkp::Ops& o, int b) { return o.add(f16[b], dc[b], jkp::Act::Selu); });
    std::vector<jkp::TensorPtr> p2 = both("avg_pool 2", [&](jkp::Ops& o, int b) { return o.avg_pool(f16[b], 2); });
    std::vector<jkp::TensorPtr> p4 = both("avg_pool 4", [&](jkp::Ops& o, int b) { return o.avg_pool(p2[b], 4); });
    both("upsample 2", [&](jkp::Ops& o, int b) { return o.upsample(p2[b], 2); });
    both("upsample 8", [&](jkp::Ops& o, int b) { return o.upsample(p4[b], 8); });
    both("concat", [&](jkp::Ops& o, int b) { return o.concat({f16[b], dc[b]}); });
    both("l2_normalize", [&](jkp::Ops& o, int b) { return o.l2_normalize(f16[b]); });
    both("l2_normalize_crop", [&](jkp::Ops& o, int b) { return o.l2_normalize_crop(f16[b], 2, 1, 61, 41); });

    // The pyramid (stored when read) and the 1x1 convolution that reads it in place.
    std::vector<jkp::TensorPtr> pyr = both("upsample_concat", [&](jkp::Ops& o, int b) {
      return o.upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8});
    });
    const jkp::Param& wp = rp.make({8, 48, 1, 1}, 0.3f);
    const jkp::Param& bp = rp.make({8}, 0.2f);
    std::vector<jkp::TensorPtr> pyr2 = both("upsample_concat (fresh)", [&](jkp::Ops& o, int b) {
      return o.upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8});
    });
    for (int b = 0; b < 2; ++b) pyr2[b] = ops[b]->upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8});
    both("conv1x1 on the pyramid 48->8", [&](jkp::Ops& o, int b) { return o.conv2d(pyr2[b], wp, &bp, jkp::Act::Selu); });
    both("l2_normalize_crop of the pyramid", [&](jkp::Ops& o, int b) {
      return o.l2_normalize_crop(o.upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8}), 3, 2, 63, 44);
    });

    // Detection and description on a score map from the network-like convolution above.
    const jkp::Param& wsc = rp.make({1, 16, 3, 3}, 0.6f);
    std::vector<jkp::TensorPtr> sm(2);
    for (int b = 0; b < 2; ++b) sm[b] = ops[b]->conv2d(f16[b], wsc, &bs, jkp::Act::Sigmoid);
    check("score map", sm[0], sm[1]);
    for (int top_k : {-1, 40}) {
      jkp::DetectParams dp;
      dp.top_k = top_k;
      dp.scores_th = 0.5f;
      jkp::Keypoints kp[2];
      for (int b = 0; b < 2; ++b) ops[b]->detect(sm[b], dp, &kp[b]);
      const std::string mode = top_k > 0 ? "top-k" : "threshold";
      check_vec("detect " + mode + " positions", kp[0].xy_norm, kp[1].xy_norm);
      check_vec("detect " + mode + " scores", kp[0].score, kp[1].score);
      jkp::SddhParams sp;
      sp.kernel = 3, sp.positions = 4;
      const int C = 48, D = 32;
      sp.offset0_w = &rp.make({8, C, 3, 3}, 0.2f);
      sp.offset0_b = &rp.make({8}, 0.2f);
      sp.offset1_w = &rp.make({8, 8, 1, 1}, 0.4f);
      sp.offset1_b = &rp.make({8}, 0.2f);
      sp.sf_w = &rp.make({C, C, 1, 1}, 0.2f);
      sp.agg_w = &rp.make({4, C, D}, 0.2f);
      std::vector<float> desc[2], desc_lazy[2];
      for (int b = 0; b < 2; ++b) {
        jkp::TensorPtr fm = ops[b]->l2_normalize(ops[b]->upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8}));
        ops[b]->describe(fm, kp[0], sp, &desc[b]);
        jkp::TensorPtr lazy = ops[b]->l2_normalize_crop(ops[b]->upsample_concat({f16[b], p2[b], p4[b]}, {1, 2, 8}), 0, 0,
                                                        72, 48);
        ops[b]->describe(lazy, kp[0], sp, &desc_lazy[b]);
      }
      check_vec("describe " + mode + " (stored map)", desc[0], desc[1]);
      check_vec("describe " + mode + " (pyramid)", desc_lazy[0], desc_lazy[1]);
    }

    // RaCo operations, on the backends that have them (the others throw from Ops::normalize).
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdev[3] = {0.229f, 0.224f, 0.225f};
    bool raco = true;
    try {
      gpu->normalize(img[1], mean, stdev);
    } catch (const std::runtime_error& e) {
      if (std::string(e.what()).find("RaCo is not implemented") == std::string::npos) throw;
      raco = false;
      std::printf("%-44s skipped (not on this backend yet)\n", "raco operations");
    }
    if (raco) {
      both("normalize", [&](jkp::Ops& o, int b) { return o.normalize(img[b], mean, stdev); });
      for (int p : {1, 2})
        both("pad_reflect " + std::to_string(p), [&](jkp::Ops& o, int b) { return o.pad_reflect(f16[b], p); });
      for (int cout : {1, 12}) {
        const jkp::Param& w5 = rp.make({cout, 16, 5, 5}, 0.2f);
        const jkp::Param& b5 = rp.make({cout}, 0.2f);
        both("conv5x5 16->" + std::to_string(cout) + " bias",
             [&](jkp::Ops& o, int b) { return o.conv2d(f16[b], w5, &b5, jkp::Act::None); });
        both("conv5x5 16->" + std::to_string(cout) + " leaky_relu",
             [&](jkp::Ops& o, int b) { return o.conv2d(f16[b], w5, nullptr, jkp::Act::LeakyRelu); });
      }
      const jkp::Param& wl = rp.make({32, 16, 3, 3}, 0.3f);
      const jkp::Param& bl = rp.make({32}, 0.2f);
      both("conv3x3 16->32 leaky_relu", [&](jkp::Ops& o, int b) { return o.conv2d(f16[b], wl, &bl, jkp::Act::LeakyRelu); });

      // Logits with a few peaks, a ranker map, raw Cholesky elements; a flat map where every pixel ties; and a
      // wide map whose logits span far more than 87, so that many probabilities are below FLT_MIN.
      const jkp::Param& wlog = rp.make({1, 16, 3, 3}, 1.5f);
      const jkp::Param& wwide = rp.make({1, 16, 3, 3}, 8.0f);
      const jkp::Param& wzero = rp.make({1, 16, 3, 3}, 0.0f);
      const jkp::Param& wrank = rp.make({1, 16, 3, 3}, 0.5f);
      const jkp::Param& wchol = rp.make({3, 16, 1, 1}, 2.0f);
      jkp::TensorPtr logits[2], flat[2], wide[2], ranker[2], chol[2];
      for (int b = 0; b < 2; ++b) {
        logits[b] = ops[b]->conv2d(f16[b], wlog, &bs, jkp::Act::None);
        wide[b] = ops[b]->conv2d(f16[b], wwide, &bs, jkp::Act::None);
        flat[b] = ops[b]->conv2d(f16[b], wzero, &bs, jkp::Act::None);
        ranker[b] = ops[b]->conv2d(f16[b], wrank, &bs, jkp::Act::None);
        chol[b] = ops[b]->conv2d(f16[b], wchol, nullptr, jkp::Act::None);
      }
      check("raco logits", logits[0], logits[1]);
      // Scores are probabilities over the whole map: both compared relative to the CPU's best one, so that a
      // wrong normalization on the GPU shows.
      auto relative = [](std::vector<float> v, const std::vector<float>& ref) {
        float m = 0;
        for (float x : ref) m = std::fmax(m, x);
        for (float& x : v) x /= m > 0 ? m : 1.0f;
        return v;
      };
      const char* map_names[3] = {"", "flat ", "wide "};
      jkp::TensorPtr* maps_by_kind[3] = {logits, flat, wide};
      for (int kind : {0, 1, 2})
        for (int top_k : {40, 100000})
          for (int window : {3, 5})
            for (bool maps : {true, false}) {
              if (kind > 0 && (window == 5 || !maps)) continue;
              // FP16 logits tie where float32 ones do not, so the set of all local maxima differs; only
              // the clear peaks of the top-40 cases are comparable.
              if (g_tol > 1e-3 && top_k > 40) continue;
              jkp::RacoDetectParams dp;
              dp.top_k = top_k, dp.window = window;
              jkp::RacoKeypoints kp[2];
              for (int b = 0; b < 2; ++b)
                ops[b]->detect_raco(maps_by_kind[kind][b], maps ? ranker[b] : nullptr, maps ? chol[b] : nullptr, dp,
                                    &kp[b]);
              const std::string mode = std::string("detect_raco ") + map_names[kind] + "top-" +
                                       std::to_string(top_k) + " " + std::to_string(window) + "x" +
                                       std::to_string(window) + (maps ? "" : " no maps");
              check_vec(mode + " positions", kp[0].xy, kp[1].xy);
              check_vec(mode + " scores", relative(kp[0].score, kp[0].score), relative(kp[1].score, kp[0].score));
              if (maps) {
                check_vec(mode + " ranker", kp[0].rank, kp[1].rank);
                check_vec(mode + " cholesky", kp[0].cholesky, kp[1].cholesky);
              } else if (!kp[1].rank.empty() || !kp[1].cholesky.empty()) {
                std::printf("%-44s FAIL (maps returned without inputs)\n", mode.c_str());
                ++g_failed;
              }
            }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all operations match\n", g_failed);
  return g_failed ? 1 : 0;
}
