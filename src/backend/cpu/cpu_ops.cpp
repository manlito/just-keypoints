// Plain C++ reference backend. Written for clarity and exact agreement with tools/folded.py;
// loops are arranged so the compiler can vectorize them, nothing more.
#include <algorithm>
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

// 5x5 max filter, window clipped at the borders (same as max_pool2d with implicit -inf padding).
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
