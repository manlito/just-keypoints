// Plain C++ reference backend. Written for clarity and exact agreement with tools/folded.py;
// loops are arranged so the compiler can vectorize them, nothing more.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "core/ops.h"

namespace jkp {
namespace {

struct CpuTensor : Tensor {
  std::vector<float> d;
  CpuTensor(int c_, int h_, int w_) : d(static_cast<size_t>(c_) * h_ * w_) { c = c_, h = h_, w = w_; }
  void read(float* dst) const override { std::memcpy(dst, d.data(), d.size() * sizeof(float)); }
  float* plane(int ch) { return d.data() + static_cast<size_t>(ch) * h * w; }
  const float* plane(int ch) const { return d.data() + static_cast<size_t>(ch) * h * w; }
};
using CpuPtr = std::shared_ptr<CpuTensor>;

const CpuTensor& cpu(const TensorPtr& t) { return static_cast<const CpuTensor&>(*t); }
CpuPtr make(int c, int h, int w) { return std::make_shared<CpuTensor>(c, h, w); }

inline float selu(float x) {
  const float alpha = 1.6732632423543772848170429916717f, scale = 1.0507009873554804934193349852946f;
  return x > 0 ? scale * x : scale * (alpha * (std::exp(x) - 1.0f));
}

void activate(float* v, int n, Act act) {
  if (act == Act::Selu) {
    for (int i = 0; i < n; ++i) v[i] = selu(v[i]);
  } else if (act == Act::Sigmoid) {
    for (int i = 0; i < n; ++i) v[i] = 1.0f / (1.0f + std::exp(-v[i]));
  } else if (act == Act::LeakyRelu) {
    for (int i = 0; i < n; ++i) v[i] = v[i] > 0 ? v[i] : 0.01f * v[i];
  } else if (act == Act::Relu) {
    for (int i = 0; i < n; ++i) v[i] = std::max(v[i], 0.0f);
  }
}

inline void axpy(float a, const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

// Bilinear sample with zeros outside the map (PyTorch grid_sample, padding_mode = zeros).
inline void bilinear_setup(float x, float y, int w, int h, int idx[4], float wt[4]) {
  float x0f = std::floor(x), y0f = std::floor(y);
  int x0 = static_cast<int>(x0f), y0 = static_cast<int>(y0f), x1 = x0 + 1, y1 = y0 + 1;
  float wx1 = x - x0f, wy1 = y - y0f, wx0 = (x0f + 1.0f) - x, wy0 = (y0f + 1.0f) - y;
  const int xs[4] = {x0, x1, x0, x1}, ys[4] = {y0, y0, y1, y1};
  const float ws[4] = {wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1};
  for (int i = 0; i < 4; ++i) {
    bool inside = xs[i] >= 0 && xs[i] < w && ys[i] >= 0 && ys[i] < h;
    idx[i] = inside ? ys[i] * w + xs[i] : 0;
    wt[i] = inside ? ws[i] : 0.0f;
  }
}
inline float bilinear_apply(const float* plane, const int idx[4], const float wt[4]) {
  return wt[0] * plane[idx[0]] + wt[1] * plane[idx[1]] + wt[2] * plane[idx[2]] + wt[3] * plane[idx[3]];
}

// PyTorch's cubic convolution weights (A = -0.75) for the taps at -1, 0, 1, 2 around floor(x), t = x - floor(x).
inline void cubic_coeffs(float t, float c[4]) {
  const float a = -0.75f;
  auto near = [a](float x) { return ((a + 2) * x - (a + 3)) * x * x + 1; };
  auto far = [a](float x) { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
  c[0] = far(t + 1.0f), c[1] = near(t), c[2] = near(1.0f - t), c[3] = far(2.0f - t);
}

// XFeat's sampling positions: pixel p of a W-wide image, normalized as 2 * (p / (W - 1)) - 1 (float32, the
// original's order) and read with align_corners = false from a map `size` wide.
inline float xfeat_source(int p, int full, int size) {
  const float g = 2.0f * (static_cast<float>(p) / static_cast<float>(full - 1)) - 1.0f;
  return ((g + 1.0f) * static_cast<float>(size) - 1.0f) / 2.0f;
}

// log(1 + exp(x)); linear above 20 as in PyTorch.
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// (2r+1)x(2r+1) max filter, window clipped at the borders (same as max_pool2d with implicit -inf padding).
void max_filter(const std::vector<float>& src, std::vector<float>& dst, int w, int h, int r) {
  std::vector<float> tmp(src.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float m = src[y * w + x];
      for (int k = std::max(0, x - r); k <= std::min(w - 1, x + r); ++k) m = std::max(m, src[y * w + k]);
      tmp[y * w + x] = m;
    }
  dst.resize(src.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float m = tmp[y * w + x];
      for (int k = std::max(0, y - r); k <= std::min(h - 1, y + r); ++k) m = std::max(m, tmp[k * w + x]);
      dst[y * w + x] = m;
    }
}

class CpuOps : public Ops {
 public:
  explicit CpuOps(int num_threads) {
    threads_ = num_threads > 0 ? num_threads : static_cast<int>(std::thread::hardware_concurrency());
    threads_ = std::max(1, threads_);
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    threads_ = 1;  // the WebAssembly build has no threads (PLAN.md, phase 5)
#endif
  }

  TensorPtr image(const uint8_t* data, int width, int height, int stride, int channels) override {
    if (channels != 1 && channels != 3 && channels != 4) throw std::runtime_error("unsupported pixel format");
    if (stride == 0) stride = width * channels;
    CpuPtr out = make(3, height, width);
    for (int c = 0; c < 3; ++c) {
      float* o = out->plane(c);
      int src_c = channels == 1 ? 0 : c;
      for (int y = 0; y < height; ++y) {
        const uint8_t* row = data + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) o[y * width + x] = static_cast<float>(row[x * channels + src_c]) / 255.0f;
      }
    }
    return out;
  }

  TensorPtr pad_replicate(const TensorPtr& xp, int left, int right, int top, int bottom) override {
    const CpuTensor& x = cpu(xp);
    CpuPtr out = make(x.c, x.h + top + bottom, x.w + left + right);
    for (int c = 0; c < x.c; ++c)
      for (int y = 0; y < out->h; ++y) {
        const float* src = x.plane(c) + std::clamp(y - top, 0, x.h - 1) * x.w;
        float* dst = out->plane(c) + y * out->w;
        for (int i = 0; i < out->w; ++i) dst[i] = src[std::clamp(i - left, 0, x.w - 1)];
      }
    return out;
  }

  TensorPtr crop(const TensorPtr& xp, int left, int top, int width, int height) override {
    const CpuTensor& x = cpu(xp);
    CpuPtr out = make(x.c, height, width);
    for (int c = 0; c < x.c; ++c)
      for (int y = 0; y < height; ++y)
        std::memcpy(out->plane(c) + y * width, x.plane(c) + (y + top) * x.w + left, width * sizeof(float));
    return out;
  }

  TensorPtr conv2d(const TensorPtr& xp, const Param& weight, const Param* bias, Act act) override {
    const CpuTensor& x = cpu(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], k = weight.shape[2], pad = k / 2;
    if (cin != x.c || weight.shape[3] != k || k % 2 == 0) throw std::runtime_error("conv2d: shape mismatch");
    CpuPtr out = make(cout, x.h, x.w);
    const int h = x.h, w = x.w;
    parallel_for(h, [&](int y_begin, int y_end) {
      for (int oc = 0; oc < cout; ++oc)
        for (int y = y_begin; y < y_end; ++y) {
          float* o = out->plane(oc) + y * w;
          std::fill(o, o + w, bias ? bias->data[oc] : 0.0f);
          for (int ic = 0; ic < cin; ++ic) {
            const float* wk = weight.data + (static_cast<size_t>(oc) * cin + ic) * k * k;
            for (int ky = 0; ky < k; ++ky) {
              int sy = y + ky - pad;
              if (sy < 0 || sy >= h) continue;
              const float* in = x.plane(ic) + sy * w;
              for (int kx = 0; kx < k; ++kx) {
                int dx = kx - pad, x0 = std::max(0, -dx), x1 = std::min(w, w - dx);
                if (x1 > x0) axpy(wk[ky * k + kx], in + x0 + dx, o + x0, x1 - x0);
              }
            }
          }
          activate(o, w, act);
        }
    });
    return out;
  }

  TensorPtr deform_conv2d(const TensorPtr& xp, const Param& offset_weight, const Param& offset_bias,
                          const Param& weight, const Param& bias, Act act) override {
    const CpuTensor& x = cpu(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], h = x.h, w = x.w;
    if (cin != x.c || weight.shape[2] != 3 || weight.shape[3] != 3 || offset_weight.shape[0] != 18)
      throw std::runtime_error("deform_conv2d: shape mismatch");
    TensorPtr offset_t = conv2d(xp, offset_weight, &offset_bias, Act::None);
    const CpuTensor& off = cpu(offset_t);
    const float max_offset = static_cast<float>(std::max(h, w)) / 4.0f;
    CpuPtr out = make(cout, h, w);
    parallel_for(h, [&](int y_begin, int y_end) {
      // col[(ic * 9 + k) * w + x]: input channel ic sampled at kernel tap k for output pixel x of this row.
      std::vector<float> col(static_cast<size_t>(cin) * 9 * w);
      for (int y = y_begin; y < y_end; ++y) {
        for (int k = 0; k < 9; ++k) {
          const float* off_y = off.plane(2 * k) + y * w;
          const float* off_x = off.plane(2 * k + 1) + y * w;
          for (int xo = 0; xo < w; ++xo) {
            float py = static_cast<float>(y - 1 + k / 3) + std::clamp(off_y[xo], -max_offset, max_offset);
            float px = static_cast<float>(xo - 1 + k % 3) + std::clamp(off_x[xo], -max_offset, max_offset);
            int idx[4] = {0, 0, 0, 0};
            float wt[4] = {0, 0, 0, 0};
            if (py > -1 && py < h && px > -1 && px < w) bilinear_setup(px, py, w, h, idx, wt);
            for (int ic = 0; ic < cin; ++ic)
              col[(static_cast<size_t>(ic) * 9 + k) * w + xo] = bilinear_apply(x.plane(ic), idx, wt);
          }
        }
        for (int oc = 0; oc < cout; ++oc) {
          float* o = out->plane(oc) + y * w;
          std::fill(o, o + w, bias.data[oc]);
          const float* wk = weight.data + static_cast<size_t>(oc) * cin * 9;
          for (int j = 0; j < cin * 9; ++j) axpy(wk[j], col.data() + static_cast<size_t>(j) * w, o, w);
          activate(o, w, act);
        }
      }
    });
    return out;
  }

  TensorPtr add(const TensorPtr& ap, const TensorPtr& bp, Act act) override {
    const CpuTensor &a = cpu(ap), &b = cpu(bp);
    if (a.d.size() != b.d.size()) throw std::runtime_error("add: shape mismatch");
    CpuPtr out = make(a.c, a.h, a.w);
    parallel_for(a.c, [&](int c0, int c1) {
      size_t plane = static_cast<size_t>(a.h) * a.w;
      for (size_t i = c0 * plane; i < c1 * plane; ++i) out->d[i] = a.d[i] + b.d[i];
      activate(out->d.data() + c0 * plane, static_cast<int>((c1 - c0) * plane), act);
    });
    return out;
  }

  TensorPtr avg_pool(const TensorPtr& xp, int k) override {
    const CpuTensor& x = cpu(xp);
    CpuPtr out = make(x.c, x.h / k, x.w / k);
    const float area = static_cast<float>(k * k);
    parallel_for(x.c, [&](int c0, int c1) {
      for (int c = c0; c < c1; ++c)
        for (int y = 0; y < out->h; ++y)
          for (int xo = 0; xo < out->w; ++xo) {
            float sum = 0;
            for (int j = 0; j < k; ++j)
              for (int i = 0; i < k; ++i) sum += x.plane(c)[(y * k + j) * x.w + xo * k + i];
            out->plane(c)[y * out->w + xo] = sum / area;
          }
    });
    return out;
  }

  TensorPtr upsample(const TensorPtr& xp, int scale) override {
    const CpuTensor& x = cpu(xp);
    const int oh = x.h * scale, ow = x.w * scale;
    CpuPtr out = make(x.c, oh, ow);
    // align_corners = true: source position = dst * (in - 1) / (out - 1).
    auto taps = [](int in, int outn, std::vector<int>& i0, std::vector<int>& i1, std::vector<float>& l1) {
      float s = outn > 1 ? static_cast<float>(in - 1) / static_cast<float>(outn - 1) : 0.0f;
      i0.resize(outn), i1.resize(outn), l1.resize(outn);
      for (int d = 0; d < outn; ++d) {
        float src = s * static_cast<float>(d);
        i0[d] = std::min(static_cast<int>(src), in - 1);
        i1[d] = i0[d] + (i0[d] < in - 1 ? 1 : 0);
        l1[d] = src - static_cast<float>(i0[d]);
      }
    };
    std::vector<int> x0, x1, y0, y1;
    std::vector<float> lx, ly;
    taps(x.w, ow, x0, x1, lx);
    taps(x.h, oh, y0, y1, ly);
    parallel_for(oh, [&](int y_begin, int y_end) {
      for (int c = 0; c < x.c; ++c)
        for (int y = y_begin; y < y_end; ++y) {
          const float* r0 = x.plane(c) + y0[y] * x.w;
          const float* r1 = x.plane(c) + y1[y] * x.w;
          float* o = out->plane(c) + y * ow;
          const float h1 = ly[y], h0 = 1.0f - h1;
          for (int i = 0; i < ow; ++i) {
            const float w1 = lx[i], w0 = 1.0f - w1;
            o[i] = h0 * (w0 * r0[x0[i]] + w1 * r0[x1[i]]) + h1 * (w0 * r1[x0[i]] + w1 * r1[x1[i]]);
          }
        }
    });
    return out;
  }

  TensorPtr concat(const std::vector<TensorPtr>& xs) override {
    int c = 0;
    for (const auto& t : xs) {
      if (t->h != xs[0]->h || t->w != xs[0]->w) throw std::runtime_error("concat: shape mismatch");
      c += t->c;
    }
    CpuPtr out = make(c, xs[0]->h, xs[0]->w);
    size_t pos = 0;
    for (const auto& t : xs) {
      const auto& d = cpu(t).d;
      std::memcpy(out->d.data() + pos, d.data(), d.size() * sizeof(float));
      pos += d.size();
    }
    return out;
  }

  TensorPtr l2_normalize(const TensorPtr& xp) override {
    const CpuTensor& x = cpu(xp);
    CpuPtr out = make(x.c, x.h, x.w);
    const int w = x.w;
    parallel_for(x.h, [&](int y_begin, int y_end) {
      std::vector<float> norm(w);
      for (int y = y_begin; y < y_end; ++y) {
        std::fill(norm.begin(), norm.end(), 0.0f);
        for (int c = 0; c < x.c; ++c) {
          const float* in = x.plane(c) + y * w;
          for (int i = 0; i < w; ++i) norm[i] += in[i] * in[i];
        }
        for (int i = 0; i < w; ++i) norm[i] = std::max(std::sqrt(norm[i]), 1e-12f);
        for (int c = 0; c < x.c; ++c) {
          const float* in = x.plane(c) + y * w;
          float* o = out->plane(c) + y * w;
          for (int i = 0; i < w; ++i) o[i] = in[i] / norm[i];
        }
      }
    });
    return out;
  }

  void detect(const TensorPtr& score_map, const DetectParams& params, Keypoints* out) override {
    const CpuTensor& sm = cpu(score_map);
    const int w = sm.w, h = sm.h, n = w * h, radius = 2;
    const std::vector<float>& s = sm.d;

    // Non-maximum suppression (simple_nms from SuperGlue, two refinement rounds).
    std::vector<float> pooled, tmp(n), nms(n);
    std::vector<char> is_max(n), suppressed(n);
    max_filter(s, pooled, w, h, radius);
    for (int i = 0; i < n; ++i) is_max[i] = s[i] == pooled[i];
    for (int round = 0; round < 2; ++round) {
      for (int i = 0; i < n; ++i) tmp[i] = is_max[i] ? 1.0f : 0.0f;
      max_filter(tmp, pooled, w, h, radius);
      for (int i = 0; i < n; ++i) {
        suppressed[i] = pooled[i] > 0;
        tmp[i] = suppressed[i] ? 0.0f : s[i];
      }
      max_filter(tmp, pooled, w, h, radius);
      for (int i = 0; i < n; ++i) is_max[i] = is_max[i] || (tmp[i] == pooled[i] && !suppressed[i]);
    }
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        bool border = x < radius || y < radius || x >= w - radius || y >= h - radius;
        nms[y * w + x] = (is_max[y * w + x] && !border) ? s[y * w + x] : 0.0f;
      }

    // Selection.
    std::vector<int> idx;
    if (params.top_k > 0) {
      idx.resize(n);
      std::iota(idx.begin(), idx.end(), 0);
      int k = std::min(params.top_k, n);
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int a, int b) { return nms[a] != nms[b] ? nms[a] > nms[b] : a < b; });
      idx.resize(k);
    } else {
      auto select = [&](float th) {
        idx.clear();
        for (int i = 0; i < n; ++i)
          if (nms[i] > th) idx.push_back(i);  // raster order
      };
      if (params.scores_th > 0) select(params.scores_th);
      if (idx.empty()) {
        float sum = 0;
        for (int i = 0; i < n; ++i) sum += s[i];
        select(sum / static_cast<float>(n));
      }
      if (static_cast<int>(idx.size()) > params.n_limit) {
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });
        idx.resize(std::max(0, params.n_limit));
      }
    }

    // Soft-argmax refinement in a 5x5 window (temperature 0.1); outside the map counts as score 0.
    const float temperature = 0.1f;
    const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
    out->xy_norm.resize(2 * idx.size());
    out->score.resize(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
      const int x = idx[i] % w, y = idx[i] / w;
      float patch[25], max_v = -INFINITY;
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          int px = x + dx, py = y + dy;
          float v = (px >= 0 && px < w && py >= 0 && py < h) ? s[py * w + px] : 0.0f;
          patch[(dy + radius) * 5 + dx + radius] = v;
          max_v = std::max(max_v, v);
        }
      float sum = 0, sx = 0, sy = 0;
      for (int j = 0; j < 25; ++j) {
        float e = std::exp((patch[j] - max_v) / temperature);
        sum += e;
        sx += e * static_cast<float>(j % 5 - radius);
        sy += e * static_cast<float>(j / 5 - radius);
      }
      // Normalized to [-1, 1]; the descriptor head truncates the pixel position recovered from this value,
      // so the float32 operation order here and in describe() follows the original exactly.
      const float xy[2] = {static_cast<float>(x) + sx / sum, static_cast<float>(y) + sy / sum};
      float* kn = &out->xy_norm[2 * i];
      for (int a = 0; a < 2; ++a) kn[a] = xy[a] / wh[a] * 2.0f - 1.0f;

      // Score: bilinear sample of the score map at the refined position.
      float gx = ((kn[0] + 1.0f) / 2.0f) * wh[0], gy = ((kn[1] + 1.0f) / 2.0f) * wh[1];
      int bi[4];
      float bw[4];
      bilinear_setup(gx, gy, w, h, bi, bw);
      out->score[i] = bilinear_apply(s.data(), bi, bw);
    }
  }

  TensorPtr normalize(const TensorPtr& xp, const float* mean, const float* std) override {
    const CpuTensor& x = cpu(xp);
    CpuPtr out = make(x.c, x.h, x.w);
    const size_t plane = static_cast<size_t>(x.h) * x.w;
    for (int c = 0; c < x.c; ++c)
      for (size_t i = 0; i < plane; ++i) out->plane(c)[i] = (x.plane(c)[i] - mean[c]) / std[c];
    return out;
  }

  TensorPtr pad_reflect(const TensorPtr& xp, int p) override {
    const CpuTensor& x = cpu(xp);
    if (p < 0 || p >= x.h || p >= x.w) throw std::runtime_error("pad_reflect: padding does not fit the tensor");
    CpuPtr out = make(x.c, x.h + 2 * p, x.w + 2 * p);
    auto mirror = [](int i, int n) { return i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i; };
    for (int c = 0; c < x.c; ++c)
      for (int y = 0; y < out->h; ++y) {
        const float* src = x.plane(c) + mirror(y - p, x.h) * x.w;
        float* dst = out->plane(c) + y * out->w;
        for (int i = 0; i < out->w; ++i) dst[i] = src[mirror(i - p, x.w)];
      }
    return out;
  }

  void detect_raco(const TensorPtr& logits_p, const TensorPtr& ranker_map, const TensorPtr& cholesky_raw,
                   const RacoDetectParams& params, RacoKeypoints* out) override {
    const CpuTensor& lm = cpu(logits_p);
    const int w = lm.w, h = lm.h, n = w * h, k = params.window, r = k / 2;
    if (lm.c != 1 || k < 1 || k % 2 == 0 || k > 9) throw std::runtime_error("detect_raco: bad arguments");
    if ((ranker_map && ranker_map->c != 1) || (cholesky_raw && cholesky_raw->c != 3))
      throw std::runtime_error("detect_raco: bad arguments");
    for (const TensorPtr& t : {ranker_map, cholesky_raw})
      if (t && (t->w != w || t->h != h)) throw std::runtime_error("detect_raco: map size mismatch");
    const std::vector<float>& logit = lm.d;

    // Probabilities: softmax over the whole image.
    std::vector<float> prob(n), pooled, nms(n);
    const float max_logit = *std::max_element(logit.begin(), logit.end());
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += prob[i] = std::exp(logit[i] - max_logit);
    const float total = static_cast<float>(sum);
    for (int i = 0; i < n; ++i) prob[i] /= total;

    // Non-maximum suppression: keep pixels equal to the maximum of their window, then the best top_k.
    // Probabilities below FLT_MIN (a logit about 87 or more below the maximum) do not count as maxima: GPUs
    // flush such denormals to zero, and the backends must agree.
    max_filter(prob, pooled, w, h, r);
    for (int i = 0; i < n; ++i) nms[i] = prob[i] == pooled[i] && prob[i] >= FLT_MIN ? prob[i] : 0.0f;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    int count = std::min(std::max(params.top_k, 0), n);
    std::partial_sort(idx.begin(), idx.begin() + count, idx.end(),
                      [&](int a, int b) { return nms[a] != nms[b] ? nms[a] > nms[b] : a < b; });
    while (count > 0 && !(nms[idx[count - 1]] > 0)) --count;  // suppressed pixels are not keypoints
    idx.resize(count);

    out->xy.resize(2 * idx.size());
    out->score.resize(idx.size());
    out->rank.resize(ranker_map ? idx.size() : 0);
    out->cholesky.resize(cholesky_raw ? 3 * idx.size() : 0);
    const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
    parallel_for(static_cast<int>(idx.size()), [&](int begin, int end) {
      std::vector<float> patch(static_cast<size_t>(k) * k);
      for (int i = begin; i < end; ++i) {
        // Subpixel refinement: softmax of the logits in the window (outside the map counts as logit 0),
        // expected offset in pixels.
        const int x = idx[i] % w, y = idx[i] / w;
        float max_v = -INFINITY;
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            int px = x + dx, py = y + dy;
            float v = ((px >= 0 && px < w && py >= 0 && py < h) ? logit[py * w + px] : 0.0f) / params.temperature;
            patch[(dy + r) * k + dx + r] = v;
            max_v = std::max(max_v, v);
          }
        float psum = 0, sx = 0, sy = 0;
        for (int j = 0; j < k * k; ++j) psum += patch[j] = std::exp(patch[j] - max_v);
        for (int j = 0; j < k * k; ++j) {
          const float p = patch[j] / psum;
          sx += p * static_cast<float>(j % k - r);
          sy += p * static_cast<float>(j / k - r);
        }
        float* xy = &out->xy[2 * static_cast<size_t>(i)];
        xy[0] = static_cast<float>(x) + sx;
        xy[1] = static_cast<float>(y) + sy;

        // Bilinear samples at the refined position, clamped to the map (grid_sample, padding_mode = border,
        // align_corners = true; same float32 steps: normalize to [-1, 1], map back, clamp).
        float g[2];
        for (int a = 0; a < 2; ++a) {
          g[a] = wh[a] > 0 ? ((2.0f * xy[a] / wh[a] - 1.0f) + 1.0f) / 2.0f * wh[a] : 0.0f;
          g[a] = std::min(wh[a], std::max(g[a], 0.0f));
        }
        int bi[4];
        float bw[4];
        bilinear_setup(g[0], g[1], w, h, bi, bw);
        out->score[i] = bilinear_apply(prob.data(), bi, bw);
        if (ranker_map) out->rank[i] = bilinear_apply(cpu(ranker_map).d.data(), bi, bw);
        if (cholesky_raw) {
          const CpuTensor& ch = cpu(cholesky_raw);
          for (int c = 0; c < 3; ++c) {
            float v[4];
            for (int j = 0; j < 4; ++j) v[j] = c == 1 ? ch.plane(c)[bi[j]] : softplus(ch.plane(c)[bi[j]]);
            const int id[4] = {0, 1, 2, 3};
            out->cholesky[3 * static_cast<size_t>(i) + c] = bilinear_apply(v, id, bw);
          }
        }
      }
    });
  }

  TensorPtr resize(const TensorPtr& xp, int oh, int ow) override {
    const CpuTensor& x = cpu(xp);
    if (oh < 1 || ow < 1) throw std::runtime_error("resize: empty output");
    CpuPtr out = make(x.c, oh, ow);
    // align_corners = false: source = max((dst + 0.5) * in / out - 0.5, 0) (PyTorch upsample_bilinear2d).
    auto taps = [](int in, int outn, std::vector<int>& i0, std::vector<int>& i1, std::vector<float>& l1) {
      const float scale = static_cast<float>(in) / static_cast<float>(outn);
      i0.resize(outn), i1.resize(outn), l1.resize(outn);
      for (int d = 0; d < outn; ++d) {
        const float src = std::max(scale * (static_cast<float>(d) + 0.5f) - 0.5f, 0.0f);
        i0[d] = std::min(static_cast<int>(src), in - 1);
        i1[d] = i0[d] + (i0[d] < in - 1 ? 1 : 0);
        l1[d] = std::min(std::max(src - static_cast<float>(i0[d]), 0.0f), 1.0f);
      }
    };
    std::vector<int> x0, x1, y0, y1;
    std::vector<float> lx, ly;
    taps(x.w, ow, x0, x1, lx);
    taps(x.h, oh, y0, y1, ly);
    parallel_for(oh, [&](int y_begin, int y_end) {
      for (int c = 0; c < x.c; ++c)
        for (int y = y_begin; y < y_end; ++y) {
          const float* r0 = x.plane(c) + y0[y] * x.w;
          const float* r1 = x.plane(c) + y1[y] * x.w;
          float* o = out->plane(c) + y * ow;
          const float h1 = ly[y], h0 = 1.0f - h1;
          for (int i = 0; i < ow; ++i) {
            const float w1 = lx[i], w0 = 1.0f - w1;
            o[i] = h0 * (w0 * r0[x0[i]] + w1 * r0[x1[i]]) + h1 * (w0 * r1[x0[i]] + w1 * r1[x1[i]]);
          }
        }
    });
    return out;
  }

  TensorPtr gray_instance_norm(const TensorPtr& xp) override {
    const CpuTensor& x = cpu(xp);
    const size_t n = static_cast<size_t>(x.h) * x.w;
    CpuPtr out = make(1, x.h, x.w);
    // Mean of the channels as PyTorch's mean (sum, then times 1/C).
    const float inv_c = 1.0f / static_cast<float>(x.c);
    for (size_t i = 0; i < n; ++i) {
      float sum = 0;
      for (int c = 0; c < x.c; ++c) sum += x.plane(c)[i];
      out->d[i] = sum * inv_c;
    }
    // Statistics in double, two passes; then x * invstd - mean * invstd in float (PyTorch's batch norm).
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += out->d[i];
    const double mean = sum / static_cast<double>(n);
    double var = 0;
    for (size_t i = 0; i < n; ++i) var += (out->d[i] - mean) * (out->d[i] - mean);
    const double invstd = 1.0 / std::sqrt(var / static_cast<double>(n) + 1e-5);
    const float alpha = static_cast<float>(invstd), beta = static_cast<float>(-mean * invstd);
    for (size_t i = 0; i < n; ++i) out->d[i] = out->d[i] * alpha + beta;
    return out;
  }

  TensorPtr conv2d_stride2(const TensorPtr& xp, const Param& weight, const Param* bias, Act act) override {
    const CpuTensor& x = cpu(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], k = weight.shape[2], pad = k / 2;
    if (cin != x.c || weight.shape[3] != k || k % 2 == 0) throw std::runtime_error("conv2d_stride2: shape mismatch");
    const int oh = (x.h + 1) / 2, ow = (x.w + 1) / 2;
    CpuPtr out = make(cout, oh, ow);
    parallel_for(oh, [&](int y_begin, int y_end) {
      for (int oc = 0; oc < cout; ++oc)
        for (int y = y_begin; y < y_end; ++y) {
          float* o = out->plane(oc) + y * ow;
          std::fill(o, o + ow, bias ? bias->data[oc] : 0.0f);
          for (int ic = 0; ic < cin; ++ic) {
            const float* wk = weight.data + (static_cast<size_t>(oc) * cin + ic) * k * k;
            for (int ky = 0; ky < k; ++ky) {
              const int sy = 2 * y + ky - pad;
              if (sy < 0 || sy >= x.h) continue;
              const float* in = x.plane(ic) + sy * x.w;
              for (int kx = 0; kx < k; ++kx) {
                const float wv = wk[ky * k + kx];
                for (int i = 0; i < ow; ++i) {
                  const int sx = 2 * i + kx - pad;
                  if (sx >= 0 && sx < x.w) o[i] += wv * in[sx];
                }
              }
            }
          }
          activate(o, ow, act);
        }
    });
    return out;
  }

  TensorPtr space_to_depth(const TensorPtr& xp, int b) override {
    const CpuTensor& x = cpu(xp);
    if (x.h % b || x.w % b) throw std::runtime_error("space_to_depth: size not a multiple of the block");
    const int oh = x.h / b, ow = x.w / b;
    CpuPtr out = make(x.c * b * b, oh, ow);
    for (int c = 0; c < x.c; ++c)
      for (int ky = 0; ky < b; ++ky)
        for (int kx = 0; kx < b; ++kx) {
          float* o = out->plane((c * b + ky) * b + kx);
          for (int y = 0; y < oh; ++y)
            for (int i = 0; i < ow; ++i) o[y * ow + i] = x.plane(c)[(y * b + ky) * x.w + i * b + kx];
        }
    return out;
  }

  void detect_xfeat(const TensorPtr& logits_p, const TensorPtr& reliability_p, const TensorPtr& feats_p,
                    const XfeatDetectParams& params, XfeatKeypoints* out) override {
    const CpuTensor &lg = cpu(logits_p), &rel = cpu(reliability_p), &f = cpu(feats_p);
    const int h8 = lg.h, w8 = lg.w, H = 8 * h8, W = 8 * w8, r = params.window / 2;
    if (lg.c != 65 || rel.c != 1 || rel.h != h8 || rel.w != w8 || f.h != h8 || f.w != w8 || params.window % 2 == 0)
      throw std::runtime_error("detect_xfeat: bad arguments");

    // Softmax over the 65 channels of each cell; the first 64 go back to their pixels.
    std::vector<float> heat(static_cast<size_t>(H) * W), pooled;
    parallel_for(h8, [&](int y_begin, int y_end) {
      float e[65];
      for (int y = y_begin; y < y_end; ++y)
        for (int x = 0; x < w8; ++x) {
          const size_t at = static_cast<size_t>(y) * w8 + x, plane = static_cast<size_t>(h8) * w8;
          float m = -INFINITY;
          for (int c = 0; c < 65; ++c) m = std::max(m, lg.d[c * plane + at]);
          float sum = 0;
          for (int c = 0; c < 65; ++c) sum += e[c] = std::exp(lg.d[c * plane + at] - m);
          for (int c = 0; c < 64; ++c) heat[static_cast<size_t>(8 * y + c / 8) * W + 8 * x + c % 8] = e[c] / sum;
        }
    });

    // NMS: local maxima of the window above the threshold, in raster order.
    max_filter(heat, pooled, W, H, r);
    std::vector<int> cand;
    for (int i = 0; i < H * W; ++i)
      if (heat[i] == pooled[i] && heat[i] > params.threshold) cand.push_back(i);

    // Score: nearest heatmap value (round half to even, 0 outside; the last row and column read outside)
    // times the bilinear reliability (zeros outside). A keypoint at (0, 0) is dropped, as in the original.
    std::vector<float> score(cand.size());
    for (size_t j = 0; j < cand.size(); ++j) {
      const int px = cand[j] % W, py = cand[j] / W;
      const int nx = static_cast<int>(std::nearbyint(xfeat_source(px, W, W)));
      const int ny = static_cast<int>(std::nearbyint(xfeat_source(py, H, H)));
      const float hv = nx >= 0 && nx < W && ny >= 0 && ny < H ? heat[static_cast<size_t>(ny) * W + nx] : 0.0f;
      int bi[4];
      float bw[4];
      bilinear_setup(xfeat_source(px, W, w8), xfeat_source(py, H, h8), w8, h8, bi, bw);
      score[j] = px == 0 && py == 0 ? -1.0f : hv * bilinear_apply(rel.d.data(), bi, bw);
    }
    std::vector<int> order(cand.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
    int count = std::min(static_cast<int>(order.size()), std::max(params.top_k, 0));
    while (count > 0 && !(score[order[count - 1]] > 0)) --count;
    order.resize(count);

    // Descriptors: bicubic samples (zeros outside) of the normalized features, normalized again.
    const int dim = f.c;
    out->dim = dim;
    out->pix.resize(2 * static_cast<size_t>(count));
    out->score.resize(count);
    out->desc.assign(static_cast<size_t>(count) * dim, 0.0f);
    parallel_for(count, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        const int px = cand[order[i]] % W, py = cand[order[i]] / W;
        out->pix[2 * i] = px, out->pix[2 * i + 1] = py;
        out->score[i] = score[order[i]];
        const float sx = xfeat_source(px, W, w8), sy = xfeat_source(py, H, h8);
        const float fx = std::floor(sx), fy = std::floor(sy);
        const int x0 = static_cast<int>(fx) - 1, y0 = static_cast<int>(fy) - 1;
        float cx[4], cy[4];
        cubic_coeffs(sx - fx, cx);
        cubic_coeffs(sy - fy, cy);
        float* d = out->desc.data() + static_cast<size_t>(i) * dim;
        for (int c = 0; c < dim; ++c) {
          const float* plane = f.plane(c);
          auto at = [&](int x, int y) { return x >= 0 && x < w8 && y >= 0 && y < h8 ? plane[y * w8 + x] : 0.0f; };
          float v = 0;
          for (int a = 0; a < 4; ++a) {
            const float row = at(x0, y0 + a) * cx[0] + at(x0 + 1, y0 + a) * cx[1] + at(x0 + 2, y0 + a) * cx[2] +
                              at(x0 + 3, y0 + a) * cx[3];
            v += row * cy[a];
          }
          d[c] = v;
        }
        float norm = 0;
        for (int c = 0; c < dim; ++c) norm += d[c] * d[c];
        norm = std::max(std::sqrt(norm), 1e-12f);
        for (int c = 0; c < dim; ++c) d[c] /= norm;
      }
    });
  }

  void describe(const TensorPtr& feature_map, const Keypoints& kpts, const SddhParams& p,
                std::vector<float>* desc) override {
    const CpuTensor& f = cpu(feature_map);
    const int c = f.c, h = f.h, w = f.w, n = kpts.n(), K = p.kernel, M = p.positions;
    const int dim = p.agg_w->shape[2], lo = (K - 1) / 2;
    if (p.offset0_w->shape[1] != c || p.agg_w->shape[1] != c || p.sf_w->shape[0] != c)
      throw std::runtime_error("describe: shape mismatch");
    const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
    const float max_offset = static_cast<float>(std::max(h, w)) / 4.0f;
    desc->assign(static_cast<size_t>(n) * dim, 0.0f);

    parallel_for(n, [&](int begin, int end) {
      std::vector<float> patch(static_cast<size_t>(c) * K * K), o0(2 * M), o1(2 * M);
      std::vector<float> sampled(static_cast<size_t>(M) * c), feat(static_cast<size_t>(M) * c);
      for (int i = begin; i < end; ++i) {
        const float kx = (kpts.xy_norm[2 * i] / 2.0f + 0.5f) * wh[0];
        const float ky = (kpts.xy_norm[2 * i + 1] / 2.0f + 0.5f) * wh[1];
        const int ix = static_cast<int>(kx), iy = static_cast<int>(ky);  // truncation

        // KxK patch centred on the truncated keypoint, zeros outside the map.
        for (int ch = 0; ch < c; ++ch)
          for (int a = 0; a < K; ++a)
            for (int b = 0; b < K; ++b) {
              int py = iy - lo + a, px = ix - lo + b;
              patch[(ch * K + a) * K + b] = (px >= 0 && px < w && py >= 0 && py < h) ? f.plane(ch)[py * w + px] : 0.0f;
            }
        // Offsets: KxK convolution (valid) -> SELU -> 1x1 convolution, clamped.
        const int taps = c * K * K;
        for (int j = 0; j < 2 * M; ++j) {
          const float* wk = p.offset0_w->data + static_cast<size_t>(j) * taps;
          float acc = 0;
          for (int t = 0; t < taps; ++t) acc += wk[t] * patch[t];
          o0[j] = selu(acc + p.offset0_b->data[j]);
        }
        for (int j = 0; j < 2 * M; ++j) {
          const float* wk = p.offset1_w->data + static_cast<size_t>(j) * 2 * M;
          float acc = 0;
          for (int t = 0; t < 2 * M; ++t) acc += wk[t] * o0[t];
          o1[j] = std::clamp(acc + p.offset1_b->data[j], -max_offset, max_offset);
        }
        // Bilinear samples of the feature map at keypoint + offset.
        for (int m = 0; m < M; ++m) {
          float gx = 2.0f * (kx + o1[m]) / wh[0] - 1.0f, gy = 2.0f * (ky + o1[M + m]) / wh[1] - 1.0f;
          gx = ((gx + 1.0f) / 2.0f) * wh[0], gy = ((gy + 1.0f) / 2.0f) * wh[1];
          int bi[4];
          float bw[4];
          bilinear_setup(gx, gy, w, h, bi, bw);
          for (int ch = 0; ch < c; ++ch) sampled[static_cast<size_t>(m) * c + ch] = bilinear_apply(f.plane(ch), bi, bw);
        }
        // 1x1 convolution + SELU on the samples, then aggregation with one CxD matrix per position.
        for (int m = 0; m < M; ++m)
          for (int oc = 0; oc < c; ++oc) {
            const float* wk = p.sf_w->data + static_cast<size_t>(oc) * c;
            const float* in = sampled.data() + static_cast<size_t>(m) * c;
            float acc = 0;
            for (int t = 0; t < c; ++t) acc += wk[t] * in[t];
            feat[static_cast<size_t>(m) * c + oc] = selu(acc);
          }
        float* d = desc->data() + static_cast<size_t>(i) * dim;
        for (int m = 0; m < M; ++m)
          for (int ch = 0; ch < c; ++ch)
            axpy(feat[static_cast<size_t>(m) * c + ch], p.agg_w->data + (static_cast<size_t>(m) * c + ch) * dim, d, dim);
        float norm = 0;
        for (int j = 0; j < dim; ++j) norm += d[j] * d[j];
        norm = std::max(std::sqrt(norm), 1e-12f);
        for (int j = 0; j < dim; ++j) d[j] /= norm;
      }
    });
  }

 private:
  // Runs fn(begin, end) over [0, n) split across the worker threads.
  void parallel_for(int n, const std::function<void(int, int)>& fn) const {
    int parts = std::min(threads_, n);
    if (parts <= 1) {
      if (n > 0) fn(0, n);
      return;
    }
    std::vector<std::thread> pool;
    for (int t = 1; t < parts; ++t)
      pool.emplace_back(fn, static_cast<int>(static_cast<long long>(n) * t / parts),
                        static_cast<int>(static_cast<long long>(n) * (t + 1) / parts));
    fn(0, n / parts);
    for (auto& th : pool) th.join();
  }

  int threads_;
};

}  // namespace

std::unique_ptr<Ops> make_cpu_ops(int num_threads) { return std::make_unique<CpuOps>(num_threads); }

}  // namespace jkp
