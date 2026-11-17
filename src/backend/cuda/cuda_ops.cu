// CUDA backend: hand-written kernels, CHW layout, same graph as the CPU backend.
//
// FP32 (the default) keeps the float32 operation order of src/backend/cpu/cpu_ops.cpp, and the file is
// compiled with --fmad=false, so the only source of difference from the CPU backend is the exp()
// implementation (SELU, sigmoid, soft-argmax).
//
// Precision: every kernel that touches a feature map is a template on its storage type R, float or
// __half. In FP16 mode feature maps and the weights of the dense network are stored as __half and
// converted to float when read; arithmetic is float32 in both modes (FP16 mode uses a fused multiply-add
// in the convolutions). The score map (the output of the sigmoid convolution) stays float so that scores
// do not tie, and so do keypoint selection and the descriptor head's own weights and buffers.
//
// Speed (PLAN.md phase 3):
// - Convolutions compute T output channels per thread, with weights stored tap-major so that the T
//   weights of a tap are adjacent. Each output accumulates in the same order as before.
// - The 128-channel full-resolution tensor of ALIKED (upsample_concat, then l2_normalize_crop) is a lazy
//   "pyramid": the 1x1 convolution of the score head and the descriptor head compute its values from
//   the four small maps where they need them. It is stored only if something else asks for its values
//   (Tensor::read in the tests, any other operation). Stored or not, FP32 results are the same.
//
// Keypoint selection (threshold, top-k, compaction, ordering) runs on the device; only the selected
// keypoints and their descriptors are copied back to the host.
//
// RaCo (normalize, pad_reflect, detect_raco) is ported from the CPU backend as on Metal (R5);
// detect_raco reuses the device-side top-k selection of detect (Selection).
//
// Profiling: with JKP_CUDA_PROFILE=1 in the environment every kernel launch is followed by a device
// synchronization and its time is recorded under the operation that issued it; the table goes to stderr
// when the backend is destroyed. Total times are meaningless in that mode, times per launch are not.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/half.h"
#include "core/ops.h"

namespace jkp {
namespace {

// ---------------------------------------------------------------------------------------------
// Device helpers shared by the kernels
// ---------------------------------------------------------------------------------------------

enum : int { kActNone = 0, kActSelu = 1, kActSigmoid = 2, kActLeakyRelu = 3 };
enum : int { kGreater = 0, kEqual = 1 };
constexpr int kRadius = 2;        // NMS and soft-argmax window radius
constexpr int kScanBlock = 256;   // elements handled by one thread in the compaction kernels
constexpr int kMaxPyramidC = 128; // channels the descriptor kernels on a pyramid can hold per thread

// Storage type <-> float.
__device__ __forceinline__ float ld(float v) { return v; }
__device__ __forceinline__ float ld(__half v) { return __half2float(v); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }

// acc + w * v. FP32 keeps the two roundings of the CPU backend (the file is compiled with --fmad=false);
// FP16 mode is not tied to the CPU backend's rounding and uses a fused multiply-add.
template <typename R>
__device__ __forceinline__ float mac(float acc, float w, float v) {
  return sizeof(R) == sizeof(float) ? acc + w * v : fmaf(w, v, acc);
}

__device__ __forceinline__ float selu(float x) {
  const float alpha = 1.6732632423543772848170429916717f, scale = 1.0507009873554804934193349852946f;
  return x > 0 ? scale * x : scale * (alpha * (expf(x) - 1.0f));
}

__device__ __forceinline__ float activate(float v, int act) {
  if (act == kActSelu) return selu(v);
  if (act == kActSigmoid) return 1.0f / (1.0f + expf(-v));
  if (act == kActLeakyRelu) return v > 0 ? v : 0.01f * v;
  return v;
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) { return v < lo ? lo : (hi < v ? hi : v); }

// Bilinear sample with zeros outside the map (PyTorch grid_sample, padding_mode = zeros).
__device__ __forceinline__ void bilinear_setup(float x, float y, int w, int h, int idx[4], float wt[4]) {
  float x0f = floorf(x), y0f = floorf(y);
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
template <typename P>
__device__ __forceinline__ float bilinear_apply(const P* plane, const int idx[4], const float wt[4]) {
  return wt[0] * ld(plane[idx[0]]) + wt[1] * ld(plane[idx[1]]) + wt[2] * ld(plane[idx[2]]) + wt[3] * ld(plane[idx[3]]);
}

__device__ __forceinline__ int thread_index() { return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); }

// ---------------------------------------------------------------------------------------------
// Dense kernels. One thread per output element unless noted.
// ---------------------------------------------------------------------------------------------

template <typename R>
__global__ void k_image(const uint8_t* src, int stride, int channels, int w, int h, R* dst) {
  int i = thread_index(), hw = h * w;
  if (i >= 3 * hw) return;
  int c = i / hw, y = (i % hw) / w, x = i % w;
  int src_c = channels == 1 ? 0 : c;
  st(dst + i, static_cast<float>(src[static_cast<size_t>(y) * stride + x * channels + src_c]) / 255.0f);
}

template <typename R>
__global__ void k_pad_replicate(const R* src, int c, int h, int w, int left, int top, int oh, int ow, R* dst) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  int sy = min(max(y - top, 0), h - 1), sx = min(max(x - left, 0), w - 1);
  dst[i] = src[(ch * h + sy) * w + sx];
}

// P: any element type (feature maps, and the float score map in FP16 mode).
template <typename P>
__global__ void k_crop(const P* src, int c, int h, int w, int left, int top, int oh, int ow, P* dst) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  dst[i] = src[(ch * h + y + top) * w + x + left];
}

// General convolution, one thread per output value; weights [cout][cin][k][k]. Used for kernel sizes
// without a tiled kernel.
template <typename R>
__global__ void k_conv2d(const R* x, const R* weight, const R* bias, int cin, int cout, int h, int w, int k, int act,
                         R* out) {
  int i = thread_index(), hw = h * w;
  if (i >= cout * hw) return;
  int oc = i / hw, y = (i % hw) / w, xo = i % w, pad = k / 2;
  float acc = bias ? ld(bias[oc]) : 0.0f;
  const R* wk = weight + static_cast<size_t>(oc) * cin * k * k;
  for (int ic = 0; ic < cin; ++ic, wk += k * k) {
    const R* in = x + static_cast<size_t>(ic) * hw;
    for (int ky = 0; ky < k; ++ky) {
      int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < k; ++kx) {
        int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        acc = mac<R>(acc, ld(wk[ky * k + kx]), ld(in[sy * w + sx]));
      }
    }
  }
  st(out + i, activate(acc, act));
}

// Weights of the tiled kernels are stored tap-major: w[tap * cpad + oc], tap = (ic * K + ky) * K + kx,
// cpad = cout rounded up to the tile size T (zeros in the padding), so the T weights one thread needs
// for a tap are adjacent.
//
// Convolution, T output channels per thread for one pixel, KxK kernel known at compile time. Each input
// value is read once for all T outputs, and each output accumulates in the order of k_conv2d, so the
// results are identical. Grid: ceil(cout / T) * h * w, pixel fastest.
// O is the output storage type: R, or float for the score map in FP16 mode.
template <typename R, typename O, int T, int K>
__global__ void k_conv2d_tile(const R* x, const R* weight, const R* bias, int cin, int cout, int h, int w, int act,
                              O* out) {
  const int hw = h * w, blocks = (cout + T - 1) / T;
  const int i = thread_index();
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, y = pix / w, xo = pix % w, pad = K / 2, oc0 = ob * T;
  float acc[T];
  #pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = bias && oc0 + t < cout ? ld(bias[oc0 + t]) : 0.0f;
  for (int ic = 0; ic < cin; ++ic) {
    const R* in = x + static_cast<size_t>(ic) * hw;
    const int wic = ic * K * K;
    for (int ky = 0; ky < K; ++ky) {
      const int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < K; ++kx) {
        const int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        const float v = ld(in[sy * w + sx]);
        const R* wp = weight + (static_cast<size_t>(wic + ky * K + kx) * blocks + ob) * T;
        #pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = mac<R>(acc[t], ld(wp[t]), v);
      }
    }
  }
  #pragma unroll
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) st(out + static_cast<size_t>(oc0 + t) * hw + pix, activate(acc[t], act));
}

// Deformable convolution, step 1. One thread per (kernel tap, pixel): samples every input channel at
// the offset position. col[(ic * 9 + tap) * hw + pixel].
template <typename R>
__global__ void k_deform_sample(const R* x, const R* off, int cin, int h, int w, float max_offset, R* col) {
  int i = thread_index(), hw = h * w;
  if (i >= 9 * hw) return;
  int k = i / hw, pix = i % hw, y = pix / w, xo = pix % w;
  float off_y = ld(off[static_cast<size_t>(2 * k) * hw + pix]), off_x = ld(off[static_cast<size_t>(2 * k + 1) * hw + pix]);
  float py = static_cast<float>(y - 1 + k / 3) + clampf(off_y, -max_offset, max_offset);
  float px = static_cast<float>(xo - 1 + k % 3) + clampf(off_x, -max_offset, max_offset);
  int idx[4] = {0, 0, 0, 0};
  float wt[4] = {0, 0, 0, 0};
  if (py > -1 && py < h && px > -1 && px < w) bilinear_setup(px, py, w, h, idx, wt);
  for (int ic = 0; ic < cin; ++ic)
    st(col + (static_cast<size_t>(ic) * 9 + k) * hw + pix, bilinear_apply(x + static_cast<size_t>(ic) * hw, idx, wt));
}

// Deformable convolution, step 2: weighted sum over the sampled columns, T output channels per thread.
// Weights are tap-major, tap = ic * 9 + kernel tap. Grid: ceil(cout / T) * hw.
template <typename R, int T>
__global__ void k_deform_sum_tile(const R* col, const R* weight, const R* bias, int taps, int cout, int hw, int act,
                                  R* out) {
  const int blocks = (cout + T - 1) / T;
  const int i = thread_index();
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, oc0 = ob * T;
  float acc[T];
  #pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = oc0 + t < cout ? ld(bias[oc0 + t]) : 0.0f;
  for (int j = 0; j < taps; ++j) {
    const float v = ld(col[static_cast<size_t>(j) * hw + pix]);
    const R* wp = weight + (static_cast<size_t>(j) * blocks + ob) * T;
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = mac<R>(acc[t], ld(wp[t]), v);
  }
  #pragma unroll
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) st(out + static_cast<size_t>(oc0 + t) * hw + pix, activate(acc[t], act));
}

template <typename R>
__global__ void k_add(const R* a, const R* b, int n, int act, R* out) {
  int i = thread_index();
  if (i >= n) return;
  st(out + i, activate(ld(a[i]) + ld(b[i]), act));
}

template <typename R>
__global__ void k_avg_pool(const R* x, int c, int w, int oh, int ow, int k, R* out) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  const R* in = x + static_cast<size_t>(ch) * (oh * k) * w;
  float sum = 0;
  for (int j = 0; j < k; ++j)
    for (int q = 0; q < k; ++q) sum += ld(in[(y * k + j) * w + xo * k + q]);
  st(out + i, sum / static_cast<float>(k * k));
}

// Bilinear, align_corners = true: source position = dst * (in - 1) / (out - 1); sx, sy are those ratios.
template <typename R>
__global__ void k_upsample(const R* x, int c, int h, int w, int oh, int ow, float sx, float sy, R* out) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  float src_x = sx * static_cast<float>(xo), src_y = sy * static_cast<float>(y);
  int x0 = min(static_cast<int>(src_x), w - 1), y0 = min(static_cast<int>(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  float w1 = src_x - static_cast<float>(x0), w0 = 1.0f - w1;
  float h1 = src_y - static_cast<float>(y0), h0 = 1.0f - h1;
  const R* r0 = x + (static_cast<size_t>(ch) * h + y0) * w;
  const R* r1 = x + (static_cast<size_t>(ch) * h + y1) * w;
  st(out + i, h0 * (w0 * ld(r0[x0]) + w1 * ld(r0[x1])) + h1 * (w0 * ld(r1[x0]) + w1 * ld(r1[x1])));
}

// L2 normalization across channels of the cropped region of x (c x h x w) at (left, top); out is
// c x oh x ow. One thread per output pixel.
template <typename R>
__global__ void k_l2_normalize_crop(const R* x, int c, int h, int w, int left, int top, int oh, int ow, R* out) {
  const int i = thread_index(), ohw = oh * ow;
  if (i >= ohw) return;
  const int hw = h * w, src = (i / ow + top) * w + i % ow + left;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = ld(x[static_cast<size_t>(ch) * hw + src]);
    norm += v * v;
  }
  norm = fmaxf(sqrtf(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch)
    st(out + static_cast<size_t>(ch) * ohw + i, ld(x[static_cast<size_t>(ch) * hw + src]) / norm);
}

// ---------------------------------------------------------------------------------------------
// Pyramid: the concatenation of up to four maps, each upsampled to the full resolution (k_upsample's
// arithmetic, or as is for scale 1). The kernels below compute its values where they are needed
// instead of storing it. In FP32 they give exactly what the stored tensor would hold.
// ---------------------------------------------------------------------------------------------

struct PyramidArgs {
  int c[4] = {0, 0, 0, 0}, h[4] = {1, 1, 1, 1}, w[4] = {1, 1, 1, 1}, scale[4] = {1, 1, 1, 1};  // c = 0: unused part
  float sx[4] = {0, 0, 0, 0}, sy[4] = {0, 0, 0, 0};                                            // k_upsample's ratios
  int H = 0, W = 0;       // full resolution
  int left = 0, top = 0;  // origin of the cropped map inside the pyramid (descriptor kernels)
};

// Where one part is read for a full-resolution pixel.
struct Taps {
  int i00, i01, i10, i11;
  float w0, w1, h0, h1;
  bool direct;
};

__device__ __forceinline__ Taps part_taps(const PyramidArgs& a, int p, int X, int Y) {
  Taps t;
  const int w = a.w[p], h = a.h[p];
  t.direct = a.scale[p] == 1;
  if (t.direct) {
    t.i00 = t.i01 = t.i10 = t.i11 = Y * w + X;
    t.w0 = t.h0 = 1.0f, t.w1 = t.h1 = 0.0f;
    return t;
  }
  float src_x = a.sx[p] * static_cast<float>(X), src_y = a.sy[p] * static_cast<float>(Y);
  int x0 = min(static_cast<int>(src_x), w - 1), y0 = min(static_cast<int>(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  t.w1 = src_x - static_cast<float>(x0), t.w0 = 1.0f - t.w1;
  t.h1 = src_y - static_cast<float>(y0), t.h0 = 1.0f - t.h1;
  t.i00 = y0 * w + x0, t.i01 = y0 * w + x1, t.i10 = y1 * w + x0, t.i11 = y1 * w + x1;
  return t;
}

// Channel `ch` of part p (stored at m) at those taps.
template <typename R>
__device__ __forceinline__ float part_value(const R* m, const PyramidArgs& a, int p, int ch, const Taps& t) {
  const R* plane = m + static_cast<size_t>(ch) * (a.h[p] * a.w[p]);
  if (t.direct) return ld(plane[t.i00]);
  return t.h0 * (t.w0 * ld(plane[t.i00]) + t.w1 * ld(plane[t.i01])) +
         t.h1 * (t.w0 * ld(plane[t.i10]) + t.w1 * ld(plane[t.i11]));
}

// 1x1 convolution on a pyramid, T output channels per thread; same accumulation order as
// k_conv2d_tile<T, 1> on the stored pyramid, tap-major weights. Grid: ceil(cout / T) * H * W.
template <typename R, typename O, int T>
__global__ void k_conv1x1_pyramid(const R* p0, const R* p1, const R* p2, const R* p3, PyramidArgs a, const R* weight,
                                  const R* bias, int cout, int act, O* out) {
  const int hw = a.H * a.W, blocks = (cout + T - 1) / T;
  const int i = thread_index();
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, Y = pix / a.W, X = pix % a.W, oc0 = ob * T;
  float acc[T];
  #pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = bias && oc0 + t < cout ? ld(bias[oc0 + t]) : 0.0f;
  const R* parts[4] = {p0, p1, p2, p3};
  int base = 0;
  for (int p = 0; p < 4; ++p) {
    if (a.c[p] == 0) continue;
    const Taps taps = part_taps(a, p, X, Y);
    for (int ch = 0; ch < a.c[p]; ++ch) {
      const float v = part_value(parts[p], a, p, ch, taps);
      const R* wp = weight + (static_cast<size_t>(base + ch) * blocks + ob) * T;
      #pragma unroll
      for (int t = 0; t < T; ++t) acc[t] = mac<R>(acc[t], ld(wp[t]), v);
    }
    base += a.c[p];
  }
  #pragma unroll
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) st(out + static_cast<size_t>(oc0 + t) * hw + pix, activate(acc[t], act));
}

// All channels of the pyramid at pixel (X, Y) into vals (at most kMaxPyramidC), and their L2 norm as
// k_l2_normalize_crop computes it.
template <typename R>
__device__ __forceinline__ float pyramid_pixel(const R* const parts[4], const PyramidArgs& a, int X, int Y,
                                               float* vals) {
  float norm = 0;
  int pos = 0;
  for (int p = 0; p < 4; ++p) {
    if (a.c[p] == 0) continue;
    const Taps taps = part_taps(a, p, X, Y);
    for (int ch = 0; ch < a.c[p]; ++ch) {
      const float v = part_value(parts[p], a, p, ch, taps);
      vals[pos++] = v;
      norm += v * v;
    }
  }
  return fmaxf(sqrtf(norm), 1e-12f);
}

// ---------------------------------------------------------------------------------------------
// Detection kernels (float in both modes)
// ---------------------------------------------------------------------------------------------

// Separable (2r+1) max filter, window clipped at the borders. dir 0 = along x, 1 = along y.
__global__ void k_max_filter(const float* src, int w, int h, int r, int dir, float* dst) {
  int i = thread_index();
  if (i >= w * h) return;
  int y = i / w, x = i % w;
  float m = src[i];
  if (dir == 0) {
    for (int k = max(0, x - r); k <= min(w - 1, x + r); ++k) m = fmaxf(m, src[y * w + k]);
  } else {
    for (int k = max(0, y - r); k <= min(h - 1, y + r); ++k) m = fmaxf(m, src[k * w + x]);
  }
  dst[i] = m;
}

// mask = 1 where the score is the maximum of its window.
__global__ void k_nms_init(const float* s, const float* pooled, int n, float* mask) {
  int i = thread_index();
  if (i >= n) return;
  mask[i] = s[i] == pooled[i] ? 1.0f : 0.0f;
}

// Scores with everything near a current maximum (pooled_mask > 0) set to zero.
__global__ void k_nms_suppress(const float* s, const float* pooled_mask, int n, float* supp_scores) {
  int i = thread_index();
  if (i >= n) return;
  supp_scores[i] = pooled_mask[i] > 0 ? 0.0f : s[i];
}

// Adds the maxima of the suppressed scores to the mask.
__global__ void k_nms_update(const float* supp_scores, const float* pooled_supp, const float* pooled_mask, int n,
                             float* mask) {
  int i = thread_index();
  if (i >= n) return;
  bool suppressed = pooled_mask[i] > 0;
  if (supp_scores[i] == pooled_supp[i] && !suppressed) mask[i] = 1.0f;
}

// nms = score at maxima away from the border, zero elsewhere.
__global__ void k_nms_final(const float* s, const float* mask, int w, int h, int r, float* nms) {
  int i = thread_index();
  if (i >= w * h) return;
  int y = i / w, x = i % w;
  bool border = x < r || y < r || x >= w - r || y >= h - r;
  nms[i] = (mask[i] > 0 && !border) ? s[i] : 0.0f;
}

// Sequential float32 sum in one thread: the mean-score fallback threshold must not depend on the
// summation order. Only runs when no score passes the configured threshold.
__global__ void k_sum_sequential(const float* v, int n, float* out) {
  if (thread_index() != 0) return;
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += v[i];
  *out = sum;
}

__device__ __forceinline__ bool selected(float v, int mode, float th) { return mode == kGreater ? v > th : v == th; }

// Ordered compaction of {i : selected(v[i])}, three steps.
// 1. One thread per block of kScanBlock elements counts the selected elements.
__global__ void k_select_count(const float* v, int n, int mode, float th, unsigned* counts) {
  int b = thread_index(), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  unsigned count = 0;
  for (int i = begin; i < end; ++i) count += selected(v[i], mode, th) ? 1u : 0u;
  counts[b] = count;
}
// 2. Exclusive prefix sum over the block counts, in place, in one thread (a few thousand blocks).
//    counts[blocks] receives the total.
__global__ void k_select_scan(unsigned* counts, int blocks) {
  if (thread_index() != 0) return;
  unsigned sum = 0;
  for (int b = 0; b < blocks; ++b) {
    unsigned c = counts[b];
    counts[b] = sum;
    sum += c;
  }
  counts[blocks] = sum;
}
// 3. One thread per block writes its indices at the block offset; only the first `limit` are kept.
__global__ void k_select_write(const float* v, int n, int mode, float th, const unsigned* offsets, unsigned limit,
                               int* out) {
  int b = thread_index(), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  unsigned pos = offsets[b];
  for (int i = begin; i < end && pos < limit; ++i)
    if (selected(v[i], mode, th)) out[pos++] = i;
}

// Radix select: histogram of 16 bits of the float bits, over the values whose higher bits equal `prefix`.
// Values are non-negative, so the bit patterns order like the floats.
__global__ void k_histogram(const float* v, int n, unsigned mask, unsigned prefix, int shift, unsigned* hist) {
  int i = thread_index();
  if (i >= n) return;
  unsigned bits = __float_as_uint(v[i]);
  if ((bits & mask) == prefix) atomicAdd(&hist[(bits >> shift) & 65535u], 1u);
}

// Rank sort of k indices by (value descending, index ascending). One thread per element counts the
// elements that come before it; the keys are distinct, so every rank is used exactly once.
__global__ void k_rank_sort(const float* v, const int* idx, int k, int* sorted) {
  int i = thread_index();
  if (i >= k) return;
  int id = idx[i];
  float vi = v[id];
  int rank = 0;
  for (int j = 0; j < k; ++j) {
    int jd = idx[j];
    float vj = v[jd];
    rank += (vj > vi || (vj == vi && jd < id)) ? 1 : 0;
  }
  sorted[rank] = id;
}

// Soft-argmax refinement in a 5x5 window (temperature 0.1); outside the map counts as score 0.
// One thread per keypoint.
__global__ void k_refine(const float* s, int w, int h, const int* idx, int n, float* xy_norm, float* score) {
  int i = thread_index();
  if (i >= n) return;
  const float temperature = 0.1f;
  const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
  const int x = idx[i] % w, y = idx[i] / w;
  float patch[25], max_v = -INFINITY;
  for (int dy = -kRadius; dy <= kRadius; ++dy)
    for (int dx = -kRadius; dx <= kRadius; ++dx) {
      int px = x + dx, py = y + dy;
      float v = (px >= 0 && px < w && py >= 0 && py < h) ? s[py * w + px] : 0.0f;
      patch[(dy + kRadius) * 5 + dx + kRadius] = v;
      max_v = fmaxf(max_v, v);
    }
  float sum = 0, sx = 0, sy = 0;
  for (int j = 0; j < 25; ++j) {
    float e = expf((patch[j] - max_v) / temperature);
    sum += e;
    sx += e * static_cast<float>(j % 5 - kRadius);
    sy += e * static_cast<float>(j / 5 - kRadius);
  }
  // Normalized to [-1, 1]; the descriptor head truncates the pixel position recovered from this value,
  // so the float32 operation order here and in the descriptor kernels follows the original exactly.
  const float xy[2] = {static_cast<float>(x) + sx / sum, static_cast<float>(y) + sy / sum};
  float kn[2];
  for (int a = 0; a < 2; ++a) kn[a] = xy[a] / wh[a] * 2.0f - 1.0f;
  xy_norm[2 * i] = kn[0];
  xy_norm[2 * i + 1] = kn[1];

  // Score: bilinear sample of the score map at the refined position.
  float gx = ((kn[0] + 1.0f) / 2.0f) * wh[0], gy = ((kn[1] + 1.0f) / 2.0f) * wh[1];
  int bi[4];
  float bw[4];
  bilinear_setup(gx, gy, w, h, bi, bw);
  score[i] = bilinear_apply(s, bi, bw);
}

// ---------------------------------------------------------------------------------------------
// Descriptor head (SDDH) kernels. n keypoints, C channels, KxK patch, M sample positions.
// Only the feature map is in the storage type R; the head's weights and buffers are float.
// ---------------------------------------------------------------------------------------------

// Offsets, first layer: KxK convolution on the patch centred on the truncated keypoint (zeros outside
// the map), then SELU. One thread per (keypoint, output channel); o0[n][2M].
template <typename R>
__global__ void k_sddh_offset0(const R* f, int c, int h, int w, const float* xy_norm, int n, int K,
                               const float* weight, const float* bias, int m2, float* o0) {
  int t = thread_index();
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2, hw = h * w, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * static_cast<float>(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * static_cast<float>(h - 1);
  const int ix = static_cast<int>(kx), iy = static_cast<int>(ky);  // truncation
  const float* wk = weight + static_cast<size_t>(j) * c * K * K;
  float acc = 0;
  for (int ch = 0; ch < c; ++ch) {
    const R* plane = f + static_cast<size_t>(ch) * hw;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) {
        int py = iy - lo + a, px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) acc += wk[(ch * K + a) * K + b] * ld(plane[py * w + px]);
      }
  }
  o0[t] = selu(acc + bias[j]);
}

// Offsets, second layer: 1x1 convolution, clamped. o1[n][2M] (x offsets first, then y).
__global__ void k_sddh_offset1(const float* o0, int n, int m2, const float* weight, const float* bias,
                               float max_offset, float* o1) {
  int t = thread_index();
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2;
  const float* wk = weight + static_cast<size_t>(j) * m2;
  const float* in = o0 + static_cast<size_t>(i) * m2;
  float acc = 0;
  for (int q = 0; q < m2; ++q) acc += wk[q] * in[q];
  o1[t] = clampf(acc + bias[j], -max_offset, max_offset);
}

// Bilinear samples of the feature map at keypoint + offset. One thread per (keypoint, position);
// sampled[n][M][C].
template <typename R>
__global__ void k_sddh_sample(const R* f, int c, int h, int w, const float* xy_norm, const float* o1, int n, int M,
                              float* sampled) {
  int t = thread_index();
  if (t >= n * M) return;
  int i = t / M, m = t % M, hw = h * w;
  const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * wh[0];
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * wh[1];
  const float* off = o1 + static_cast<size_t>(i) * 2 * M;
  float gx = 2.0f * (kx + off[m]) / wh[0] - 1.0f, gy = 2.0f * (ky + off[M + m]) / wh[1] - 1.0f;
  gx = ((gx + 1.0f) / 2.0f) * wh[0], gy = ((gy + 1.0f) / 2.0f) * wh[1];
  int bi[4];
  float bw[4];
  bilinear_setup(gx, gy, w, h, bi, bw);
  float* out = sampled + static_cast<size_t>(t) * c;
  for (int ch = 0; ch < c; ++ch) out[ch] = bilinear_apply(f + static_cast<size_t>(ch) * hw, bi, bw);
}

// 1x1 convolution + SELU on the samples, T output channels per thread; weights tap-major (input channel
// major). One thread per (keypoint, position, block of T output channels). R only selects mac().
template <typename R, int T>
__global__ void k_sddh_sf_tile(const float* sampled, int total, int c, const float* weight, float* feat) {
  const int blocks = (c + T - 1) / T, i = thread_index();
  if (i >= total / c * blocks) return;
  const int s = i / blocks, ob = i % blocks, oc0 = ob * T;
  const float* in = sampled + static_cast<size_t>(s) * c;
  float acc[T];
  #pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = 0.0f;
  for (int ch = 0; ch < c; ++ch) {
    const float v = in[ch];
    const float* wp = weight + (static_cast<size_t>(ch) * blocks + ob) * T;
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = mac<R>(acc[t], wp[t], v);
  }
  #pragma unroll
  for (int t = 0; t < T; ++t)
    if (oc0 + t < c) feat[static_cast<size_t>(s) * c + oc0 + t] = selu(acc[t]);
}

// Aggregation with one CxD matrix per position, T descriptor elements per thread (the last block may be
// shorter). One thread per (keypoint, block of T elements); weight[M*C][dim].
template <typename R, int T>
__global__ void k_sddh_aggregate_tile(const float* feat, int n, int mc, int dim, const float* weight, float* desc) {
  const int blocks = (dim + T - 1) / T, t = thread_index();
  if (t >= n * blocks) return;
  const int i = t / blocks, j0 = (t % blocks) * T;
  const float* in = feat + static_cast<size_t>(i) * mc;
  float acc[T];
  #pragma unroll
  for (int q = 0; q < T; ++q) acc[q] = 0.0f;
  for (int k = 0; k < mc; ++k) {
    const float v = in[k];
    const float* wp = weight + static_cast<size_t>(k) * dim + j0;
    #pragma unroll
    for (int q = 0; q < T; ++q)
      if (j0 + q < dim) acc[q] = mac<R>(acc[q], v, wp[q]);
  }
  #pragma unroll
  for (int q = 0; q < T; ++q)
    if (j0 + q < dim) desc[static_cast<size_t>(i) * dim + j0 + q] = acc[q];
}

// L2 normalization of each descriptor. One thread per keypoint.
__global__ void k_normalize_rows(float* desc, int n, int dim) {
  int i = thread_index();
  if (i >= n) return;
  float* d = desc + static_cast<size_t>(i) * dim;
  float norm = 0;
  for (int j = 0; j < dim; ++j) norm += d[j] * d[j];
  norm = fmaxf(sqrtf(norm), 1e-12f);
  for (int j = 0; j < dim; ++j) d[j] /= norm;
}

// Descriptor head on a pyramid: the feature map is the L2-normalized pyramid, cropped to w x h at
// (a.left, a.top). These kernels normalize only the pixels the keypoints touch; they replace
// k_sddh_offset0 and k_sddh_sample and give the same values as those do on the stored feature map.

// Normalized KxK patch around the truncated keypoint. One thread per (keypoint, patch pixel);
// patch[n][K*K][C], left untouched (and unused) for pixels outside the map.
template <typename R>
__global__ void k_sddh_patch_pyramid(const R* p0, const R* p1, const R* p2, const R* p3, PyramidArgs a, int h, int w,
                                     const float* xy_norm, int n, int K, float* patch) {
  const int t = thread_index(), kk = K * K;
  if (t >= n * kk) return;
  const int i = t / kk, q = t % kk, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * static_cast<float>(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * static_cast<float>(h - 1);
  const int px = static_cast<int>(kx) - lo + q % K, py = static_cast<int>(ky) - lo + q / K;  // truncation
  if (px < 0 || px >= w || py < 0 || py >= h) return;
  const R* parts[4] = {p0, p1, p2, p3};
  const int c = a.c[0] + a.c[1] + a.c[2] + a.c[3];
  float vals[kMaxPyramidC];
  const float norm = pyramid_pixel(parts, a, px + a.left, py + a.top, vals);
  float* out = patch + static_cast<size_t>(t) * c;
  for (int ch = 0; ch < c; ++ch) out[ch] = vals[ch] / norm;
}

// k_sddh_offset0 on the patches of k_sddh_patch_pyramid.
__global__ void k_sddh_offset0_patch(const float* patch, int c, int h, int w, const float* xy_norm, int n, int K,
                                     const float* weight, const float* bias, int m2, float* o0) {
  int t = thread_index();
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * static_cast<float>(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * static_cast<float>(h - 1);
  const int ix = static_cast<int>(kx), iy = static_cast<int>(ky);  // truncation
  const float* wk = weight + static_cast<size_t>(j) * c * K * K;
  const float* in = patch + static_cast<size_t>(i) * K * K * c;
  float acc = 0;
  for (int ch = 0; ch < c; ++ch)
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) {
        int py = iy - lo + a, px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) acc += wk[(ch * K + a) * K + b] * in[(a * K + b) * c + ch];
      }
  o0[t] = selu(acc + bias[j]);
}

// k_sddh_sample on the pyramid: the four pixels around each sample position are normalized here.
// One thread per (keypoint, position); sampled[n][M][C], C <= kMaxPyramidC.
template <typename R>
__global__ void k_sddh_sample_pyramid(const R* p0, const R* p1, const R* p2, const R* p3, PyramidArgs a, int h, int w,
                                      const float* xy_norm, const float* o1, int n, int M, float* sampled) {
  int t = thread_index();
  if (t >= n * M) return;
  int i = t / M, m = t % M;
  const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * wh[0];
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * wh[1];
  const float* off = o1 + static_cast<size_t>(i) * 2 * M;
  float gx = 2.0f * (kx + off[m]) / wh[0] - 1.0f, gy = 2.0f * (ky + off[M + m]) / wh[1] - 1.0f;
  gx = ((gx + 1.0f) / 2.0f) * wh[0], gy = ((gy + 1.0f) / 2.0f) * wh[1];
  int bi[4];
  float bw[4];
  bilinear_setup(gx, gy, w, h, bi, bw);
  const R* parts[4] = {p0, p1, p2, p3};
  const int c = a.c[0] + a.c[1] + a.c[2] + a.c[3];
  float vals[kMaxPyramidC], acc[kMaxPyramidC];
  // A corner outside the map has weight 0 and index 0, as in bilinear_apply: it adds 0 * (pixel 0).
  for (int k = 0; k < 4; ++k) {
    const float norm = pyramid_pixel(parts, a, bi[k] % w + a.left, bi[k] / w + a.top, vals);
    for (int ch = 0; ch < c; ++ch) {
      const float v = bw[k] * (vals[ch] / norm);
      acc[ch] = k == 0 ? v : acc[ch] + v;
    }
  }
  float* out = sampled + static_cast<size_t>(t) * c;
  for (int ch = 0; ch < c; ++ch) out[ch] = acc[ch];
}

// ---------------------------------------------------------------------------------------------
// RaCo kernels (normalize, pad_reflect, detect_raco). Same float32 steps as the CPU backend, except
// that the softmax sum is a double sum in blocks instead of a double sum in raster order.
// ---------------------------------------------------------------------------------------------

// (x - mean[c]) / std[c]; ms holds the c means, then the c standard deviations.
template <typename R>
__global__ void k_normalize(const R* x, int c, int hw, const float* ms, R* out) {
  int i = thread_index();
  if (i >= c * hw) return;
  int ch = i / hw;
  st(out + i, (ld(x[i]) - ms[ch]) / ms[c + ch]);
}

// Mirror padding by p on every side, without repeating the border pixel (PyTorch "reflect").
__device__ __forceinline__ int mirror(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i); }
template <typename R>
__global__ void k_pad_reflect(const R* src, int c, int h, int w, int p, R* dst) {
  const int oh = h + 2 * p, ow = w + 2 * p, ohw = oh * ow;
  int i = thread_index();
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  dst[i] = src[(static_cast<size_t>(ch) * h + mirror(y - p, h)) * w + mirror(x - p, w)];
}

// Softmax over the whole map: block maxima, the maximum (stats[0]); block sums of exp(l - max) in double,
// their sum as float (stats[1]); then the probabilities. One thread per block of kScanBlock elements for
// the block steps, one thread for the reductions. The double sums keep the result as close to the CPU
// backend's raster-order double sum as float32 can show.
template <typename R>
__global__ void k_softmax_block_max(const R* l, int n, float* partial) {
  int b = thread_index(), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  float m = ld(l[begin]);
  for (int i = begin + 1; i < end; ++i) m = fmaxf(m, ld(l[i]));
  partial[b] = m;
}
__global__ void k_softmax_max(const float* partial, int blocks, float* stats) {
  if (thread_index() != 0) return;
  float m = partial[0];
  for (int b = 1; b < blocks; ++b) m = fmaxf(m, partial[b]);
  stats[0] = m;
}
template <typename R>
__global__ void k_softmax_block_sum(const R* l, int n, const float* stats, double* partial) {
  int b = thread_index(), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  const float m = stats[0];
  double sum = 0;
  for (int i = begin; i < end; ++i) sum += static_cast<double>(expf(ld(l[i]) - m));
  partial[b] = sum;
}
__global__ void k_softmax_sum(const double* partial, int blocks, float* stats) {
  if (thread_index() != 0) return;
  double sum = 0;
  for (int b = 0; b < blocks; ++b) sum += partial[b];
  stats[1] = static_cast<float>(sum);
}
template <typename R>
__global__ void k_softmax_prob(const R* l, int n, const float* stats, float* prob) {
  int i = thread_index();
  if (i >= n) return;
  prob[i] = expf(ld(l[i]) - stats[0]) / stats[1];
}

// nms = probability where it equals the maximum of its window, zero elsewhere (no border exclusion).
// Probabilities below FLT_MIN are not maxima, as on the CPU: GPUs may flush them to zero.
__global__ void k_raco_nms(const float* prob, const float* pooled, int n, float* nms) {
  int i = thread_index();
  if (i >= n) return;
  nms[i] = prob[i] == pooled[i] && prob[i] >= FLT_MIN ? prob[i] : 0.0f;
}

// log(1 + exp(x)); linear above 20 as in PyTorch.
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Subpixel refinement and sampling, one thread per keypoint: softmax of logit / temperature in the k x k
// window (outside the map counts as logit 0), expected offset; then bilinear samples at the refined
// position, clamped to the map, of the probabilities, the ranker map and the Cholesky map (softplus on
// channels 0 and 2 at each of the four neighbours). ranker and chol may be null.
template <typename R>
__global__ void k_raco_refine(const R* logits, const float* prob, const R* ranker, const R* chol, int w, int h,
                              const int* idx, int n, int k, float temperature, float* xy_out, float* score,
                              float* rank, float* chol_out) {
  int i = thread_index();
  if (i >= n) return;
  const int r = k / 2, x = idx[i] % w, y = idx[i] / w;
  float patch[81], max_v = -INFINITY;
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx) {
      int px = x + dx, py = y + dy;
      float v = ((px >= 0 && px < w && py >= 0 && py < h) ? ld(logits[py * w + px]) : 0.0f) / temperature;
      patch[(dy + r) * k + dx + r] = v;
      max_v = fmaxf(max_v, v);
    }
  float psum = 0, sx = 0, sy = 0;
  for (int j = 0; j < k * k; ++j) {
    patch[j] = expf(patch[j] - max_v);
    psum += patch[j];
  }
  for (int j = 0; j < k * k; ++j) {
    const float p = patch[j] / psum;
    sx += p * static_cast<float>(j % k - r);
    sy += p * static_cast<float>(j / k - r);
  }
  const float xy[2] = {static_cast<float>(x) + sx, static_cast<float>(y) + sy};
  xy_out[2 * i] = xy[0];
  xy_out[2 * i + 1] = xy[1];

  // grid_sample, padding_mode = border, align_corners = true: normalize to [-1, 1], map back, clamp.
  const float wh[2] = {static_cast<float>(w - 1), static_cast<float>(h - 1)};
  float g[2];
  for (int a = 0; a < 2; ++a) {
    g[a] = wh[a] > 0 ? ((2.0f * xy[a] / wh[a] - 1.0f) + 1.0f) / 2.0f * wh[a] : 0.0f;
    g[a] = fminf(wh[a], fmaxf(g[a], 0.0f));
  }
  int bi[4];
  float bw[4];
  bilinear_setup(g[0], g[1], w, h, bi, bw);
  score[i] = bilinear_apply(prob, bi, bw);
  if (ranker) rank[i] = bilinear_apply(ranker, bi, bw);
  if (chol) {
    const int hw = w * h;
    for (int c = 0; c < 3; ++c) {
      const R* plane = chol + static_cast<size_t>(c) * hw;
      float v[4];
      for (int j = 0; j < 4; ++j) v[j] = c == 1 ? ld(plane[bi[j]]) : softplus(ld(plane[bi[j]]));
      chol_out[3 * i + c] = bw[0] * v[0] + bw[1] * v[1] + bw[2] * v[2] + bw[3] * v[3];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(status));
}
#define JKP_CUDA_CHECK(call) check((call), #call)

constexpr int kBlock = 256;

// JKP_CUDA_PROFILE: time per operation label (see the top of the file).
struct Profiler {
  bool on = false;
  long frames = 0;
  std::string label;
  std::map<std::string, std::pair<double, long>> stats;  // label -> (ms, launches)

  Profiler() {
    const char* env = std::getenv("JKP_CUDA_PROFILE");
    on = env && *env && std::string(env) != "0";
  }
  void tag(const char* fmt, ...) {
    if (!on) return;
    char buf[160];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    label = buf;
  }
  void report() {
    if (!on || stats.empty()) return;
    const double per = static_cast<double>(std::max(1L, frames));
    double total = 0;
    std::vector<std::pair<std::string, std::pair<double, long>>> rows(stats.begin(), stats.end());
    for (const auto& r : rows) total += r.second.first;
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    std::fprintf(stderr, "\njkp cuda profile: %ld frames, kernel time %.2f ms/frame\n\n%-44s %10s %7s %6s\n", frames,
                 total / per, "operation", "ms/frame", "share", "calls");
    for (const auto& r : rows)
      std::fprintf(stderr, "%-44s %10.3f %6.1f%% %6.1f\n", r.first.c_str(), r.second.first / per,
                   100.0 * r.second.first / total, r.second.second / per);
    stats.clear();
  }
};
Profiler g_profile;

template <typename T>
struct Identity {
  using type = T;
};

// Launches `kernel` with at least n threads. The kernel's parameter types fix the argument types.
template <typename... Args>
void launch(void (*kernel)(Args...), long long n, typename Identity<Args>::type... args) {
  if (n <= 0) return;
  if (n > static_cast<long long>(INT_MAX)) throw std::runtime_error("cuda: grid too large");
  std::chrono::steady_clock::time_point t0;
  if (g_profile.on) {
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
  }
  kernel<<<(static_cast<unsigned>(n) + kBlock - 1) / kBlock, kBlock>>>(args...);
  check(cudaGetLastError(), "kernel launch");
  if (g_profile.on) {
    cudaDeviceSynchronize();
    auto& st = g_profile.stats[g_profile.label];
    st.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++st.second;
  }
}

// Element count as int; the kernels index with 32-bit integers.
int count(long long n) {
  if (n < 0 || n > INT_MAX) throw std::runtime_error("cuda: tensor too large");
  return static_cast<int>(n);
}

float upsample_ratio(int in, int out) { return out > 1 ? static_cast<float>(in - 1) / static_cast<float>(out - 1) : 0.0f; }

// Device memory with reuse: released blocks are kept and handed out again for the same size, so a
// stream of equally sized images allocates only on the first frame.
class Pool {
 public:
  explicit Pool(int device) : device_(device) {}
  ~Pool() {
    if (cudaSetDevice(device_) == cudaSuccess) trim();
  }
  int device() const { return device_; }

  void* alloc(size_t bytes) {
    bytes = std::max<size_t>(bytes, 1);
    auto it = free_.find(bytes);
    if (it != free_.end()) {
      void* p = it->second;
      free_.erase(it);
      return p;
    }
    void* p = nullptr;
    cudaError_t status = cudaMalloc(&p, bytes);
    if (status == cudaErrorMemoryAllocation) {  // give back what is cached and try once more
      cudaGetLastError();
      trim();
      status = cudaMalloc(&p, bytes);
    }
    check(status, "cudaMalloc");
    return p;
  }
  void release(void* p, size_t bytes) { free_.emplace(std::max<size_t>(bytes, 1), p); }
  void trim() {
    for (auto& kv : free_) cudaFree(kv.second);
    free_.clear();
  }

 private:
  int device_;
  std::multimap<size_t, void*> free_;
};
using PoolPtr = std::shared_ptr<Pool>;

// A typed block from the pool, returned to it on destruction.
template <typename T>
class Buffer {
 public:
  Buffer(const PoolPtr& pool, size_t n) : pool_(pool), n_(n), p_(static_cast<T*>(pool->alloc(n * sizeof(T)))) {}
  ~Buffer() { pool_->release(p_, n_ * sizeof(T)); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() const { return p_; }
  size_t size() const { return n_; }
  void upload(const T* src, size_t n) {
    if (n) JKP_CUDA_CHECK(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* dst, size_t n) const {
    if (n) JKP_CUDA_CHECK(cudaMemcpy(dst, p_, n * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  PoolPtr pool_;
  size_t n_;
  T* p_;
};

// Ordered selection on a device map of n non-negative floats. Counting is a round trip to the host.
class Selection {
 public:
  Selection(const PoolPtr& pool, const float* v, int n)
      : pool_(pool), v_(v), n_(n), blocks_((n + kScanBlock - 1) / kScanBlock), offsets_(pool, static_cast<size_t>(blocks_) + 1) {}

  // Number of {v `mode` th}; leaves the per-block offsets for write().
  int count(int mode, float th) {
    launch(k_select_count, blocks_, v_, n_, mode, th, offsets_.get());
    launch(k_select_scan, 1, offsets_.get(), blocks_);
    unsigned total = 0;
    JKP_CUDA_CHECK(cudaMemcpy(&total, offsets_.get() + blocks_, sizeof(total), cudaMemcpyDeviceToHost));
    return static_cast<int>(total);
  }
  // Writes the first `limit` of that set in raster order; must follow the matching count().
  void write(int mode, float th, int limit, int* dst) {
    launch(k_select_write, blocks_, v_, n_, mode, th, offsets_.get(), static_cast<unsigned>(limit), dst);
  }
  // Indices of the `best` (<= n) highest values, best first, ties in raster order.
  void top(int best, int* dst) {
    // Radix select for the value of the `best`-th highest, 16 bits of the float bits per pass (each pass
    // is a round trip to the host).
    const int bins = 65536;
    Buffer<unsigned> hist_dev(pool_, bins);
    std::vector<unsigned> hist(bins);
    unsigned prefix = 0, mask = 0;
    int remaining = best;  // how many are still needed from the values matching the prefix
    for (int shift = 16; shift >= 0; shift -= 16) {
      JKP_CUDA_CHECK(cudaMemset(hist_dev.get(), 0, bins * sizeof(unsigned)));
      launch(k_histogram, n_, v_, n_, mask, prefix, shift, hist_dev.get());
      hist_dev.download(hist.data(), bins);
      int bin = bins - 1;
      for (; bin > 0 && static_cast<long long>(hist[bin]) < remaining; --bin) remaining -= static_cast<int>(hist[bin]);
      prefix |= static_cast<unsigned>(bin) << shift;
      mask |= 65535u << shift;
    }
    float kth;
    std::memcpy(&kth, &prefix, sizeof(kth));
    // Everything above that value, then the first `remaining` equal to it in raster order.
    const int above = best - remaining;
    Buffer<int> unsorted(pool_, best);
    if (count(kGreater, kth) != above) throw std::runtime_error("cuda: top-k selection is inconsistent");
    write(kGreater, kth, above, unsorted.get());
    count(kEqual, kth);
    write(kEqual, kth, remaining, unsorted.get() + above);
    g_profile.tag("rank sort");
    launch(k_rank_sort, best, v_, unsorted.get(), best, dst);
  }

 private:
  PoolPtr pool_;
  const float* v_;
  int n_, blocks_;
  Buffer<unsigned> offsets_;
};

struct CudaTensor;
using CudaPtr = std::shared_ptr<CudaTensor>;

// concat of parts[i] upsampled by scales[i]; with `normalized`, L2-normalized across channels and
// cropped to the tensor's size at (left, top).
struct Pyramid {
  std::vector<CudaPtr> parts;
  std::vector<int> scales;
  int height = 0, width = 0;  // full resolution
  bool normalized = false;
  int left = 0, top = 0;
  void (*store)(const CudaTensor&, void* dst) = nullptr;  // computes the tensor's values into dst
};

// Elements are float, or __half when the backend runs in FP16 and the tensor is not the score map.
// A tensor with a pyramid has no storage until data() or read() asks for it.
struct CudaTensor : Tensor {
  PoolPtr pool;
  bool half;
  int n;
  std::shared_ptr<const Pyramid> pyramid;
  CudaTensor(const PoolPtr& pool_, int c_, int h_, int w_, bool half_, std::shared_ptr<const Pyramid> pyramid_ = {})
      : pool(pool_), half(half_), n(count(static_cast<long long>(c_) * h_ * w_)), pyramid(std::move(pyramid_)) {
    c = c_, h = h_, w = w_;
    if (!pyramid) buf_ = std::make_unique<Buffer<uint8_t>>(pool, bytes());
  }
  size_t bytes() const { return static_cast<size_t>(n) * (half ? sizeof(__half) : sizeof(float)); }
  void read(float* dst) const override {
    JKP_CUDA_CHECK(cudaSetDevice(pool->device()));
    const void* src = data();
    if (!n) return;
    if (!half) {
      JKP_CUDA_CHECK(cudaMemcpy(dst, src, bytes(), cudaMemcpyDeviceToHost));
      return;
    }
    std::vector<uint16_t> bits(n);
    JKP_CUDA_CHECK(cudaMemcpy(bits.data(), src, bytes(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) dst[i] = half_to_float(bits[i]);
  }
  void* data() const {
    if (!buf_) {
      auto out = std::make_unique<Buffer<uint8_t>>(pool, bytes());
      pyramid->store(*this, out->get());
      buf_ = std::move(out);
    }
    return buf_->get();
  }
  template <typename T>
  const T* as() const {
    if ((sizeof(T) == sizeof(__half)) != half) throw std::runtime_error("cuda: tensor has another element type");
    return static_cast<const T*>(data());
  }
  template <typename T>
  T* out() {
    return const_cast<T*>(as<T>());
  }
  int size() const { return n; }
  PyramidArgs pyramid_args() const {
    PyramidArgs a;
    const Pyramid& p = *pyramid;
    for (size_t i = 0; i < p.parts.size(); ++i) {
      const CudaTensor& x = *p.parts[i];
      a.c[i] = x.c, a.h[i] = x.h, a.w[i] = x.w, a.scale[i] = p.scales[i];
      a.sx[i] = upsample_ratio(x.w, p.width), a.sy[i] = upsample_ratio(x.h, p.height);
    }
    a.H = p.height, a.W = p.width, a.left = p.left, a.top = p.top;
    return a;
  }

 private:
  mutable std::unique_ptr<Buffer<uint8_t>> buf_;
};

const CudaTensor& dev(const TensorPtr& t) { return static_cast<const CudaTensor&>(*t); }

int act_id(Act act) {
  switch (act) {
    case Act::None: return kActNone;
    case Act::Selu: return kActSelu;
    case Act::Sigmoid: return kActSigmoid;
    case Act::LeakyRelu: return kActLeakyRelu;
  }
  throw std::runtime_error("cuda: unknown activation");
}

// Bit pattern of a host float in the storage type.
inline void encode(float v, float* dst) { *dst = v; }
inline void encode(float v, __half* dst) {
  const uint16_t bits = float_to_half(v);
  std::memcpy(dst, &bits, sizeof(bits));
}

// R: storage type of feature maps and of the dense network's weights (float or __half).
template <typename R>
class CudaOps : public Ops {
  static constexpr bool kHalf = sizeof(R) == sizeof(__half);

 public:
  explicit CudaOps(int device) {
    int devices = 0;
    JKP_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (device < 0 || device >= devices)
      throw std::runtime_error("cuda: device " + std::to_string(device) + " not found (" + std::to_string(devices) +
                               " available)");
    JKP_CUDA_CHECK(cudaSetDevice(device));
    pool_ = std::make_shared<Pool>(device);
  }
  ~CudaOps() override {
    if (cudaSetDevice(pool_->device()) == cudaSuccess)
      for (void* p : params_owned_) cudaFree(p);
    g_profile.report();
  }

  TensorPtr image(const uint8_t* data, int width, int height, int stride, int channels) override {
    use();
    if (channels != 1 && channels != 3 && channels != 4) throw std::runtime_error("unsupported pixel format");
    if (stride == 0) stride = width * channels;
    CudaPtr out = make(3, height, width);
    // The last row may be shorter than the stride.
    size_t bytes = static_cast<size_t>(stride) * (height - 1) + static_cast<size_t>(width) * channels;
    Buffer<uint8_t> src(pool_, bytes);
    src.upload(data, bytes);
    ++g_profile.frames;
    g_profile.tag("image %dx%d", width, height);
    launch(k_image<R>, out->size(), src.get(), stride, channels, width, height, out->template out<R>());
    return out;
  }

  TensorPtr pad_replicate(const TensorPtr& xp, int left, int right, int top, int bottom) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h + top + bottom, x.w + left + right);
    g_profile.tag("pad %dx%d", out->w, out->h);
    launch(k_pad_replicate<R>, out->size(), x.as<R>(), x.c, x.h, x.w, left, top, out->h, out->w, out->template out<R>());
    return out;
  }

  TensorPtr crop(const TensorPtr& xp, int left, int top, int width, int height) override {
    use();
    const CudaTensor& x = dev(xp);
    if (left == 0 && top == 0 && width == x.w && height == x.h) return xp;  // tensors are never modified
    CudaPtr out = std::make_shared<CudaTensor>(pool_, x.c, height, width, x.half);
    g_profile.tag("crop c%d %dx%d", x.c, width, height);
    if (x.half)
      launch(k_crop<__half>, out->size(), x.as<__half>(), x.c, x.h, x.w, left, top, height, width, out->template out<__half>());
    else
      launch(k_crop<float>, out->size(), x.as<float>(), x.c, x.h, x.w, left, top, height, width, out->template out<float>());
    return out;
  }

  TensorPtr conv2d(const TensorPtr& xp, const Param& weight, const Param* bias, Act act) override {
    use();
    const CudaTensor& x = dev(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], k = weight.shape[2];
    if (cin != x.c || weight.shape[3] != k || k % 2 == 0) throw std::runtime_error("conv2d: shape mismatch");
    const R* b = bias ? param(*bias) : nullptr;
    const int hw = x.h * x.w, a = act_id(act);
    g_profile.tag("conv%dx%d %3d->%-3d %4dx%-4d%s", k, k, cin, cout, x.w, x.h, x.pyramid && k == 1 ? " pyramid" : "");
    // FP16: the sigmoid layer produces the score map, which stays float (see the top of the file).
    if (kHalf && act == Act::Sigmoid && cout <= 4 && (k == 1 || k == 3)) {
      CudaPtr out = std::make_shared<CudaTensor>(pool_, cout, x.h, x.w, false);
      const R* w = param_tiled(weight, 4);
      if (k == 1) launch(k_conv2d_tile<R, float, 4, 1>, hw, x.as<R>(), w, b, cin, cout, x.h, x.w, a, out->template out<float>());
      else launch(k_conv2d_tile<R, float, 4, 3>, hw, x.as<R>(), w, b, cin, cout, x.h, x.w, a, out->template out<float>());
      return out;
    }
    CudaPtr out = make(cout, x.h, x.w);
    R* o = out->template out<R>();
    const int t = tile(cout);
    const long long threads = static_cast<long long>((cout + t - 1) / t) * hw;
    if (k == 1 && x.pyramid && !x.pyramid->normalized) {
      const PyramidArgs args = x.pyramid_args();
      const R* w = param_tiled(weight, t);
      const R *p0 = part(x, 0), *p1 = part(x, 1), *p2 = part(x, 2), *p3 = part(x, 3);
      if (t == 4) launch(k_conv1x1_pyramid<R, R, 4>, threads, p0, p1, p2, p3, args, w, b, cout, a, o);
      else if (t == 8) launch(k_conv1x1_pyramid<R, R, 8>, threads, p0, p1, p2, p3, args, w, b, cout, a, o);
      else launch(k_conv1x1_pyramid<R, R, 16>, threads, p0, p1, p2, p3, args, w, b, cout, a, o);
    } else if (k == 1 || k == 3 || k == 5) {  // 5: RaCo's ranker output
      const R* w = param_tiled(weight, t);
      const R* in = x.as<R>();
      if (k == 1 && t == 4) launch(k_conv2d_tile<R, R, 4, 1>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (k == 1 && t == 8) launch(k_conv2d_tile<R, R, 8, 1>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (k == 1) launch(k_conv2d_tile<R, R, 16, 1>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (k == 3 && t == 4) launch(k_conv2d_tile<R, R, 4, 3>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (k == 3 && t == 8) launch(k_conv2d_tile<R, R, 8, 3>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (k == 3) launch(k_conv2d_tile<R, R, 16, 3>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (t == 4) launch(k_conv2d_tile<R, R, 4, 5>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else if (t == 8) launch(k_conv2d_tile<R, R, 8, 5>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
      else launch(k_conv2d_tile<R, R, 16, 5>, threads, in, w, b, cin, cout, x.h, x.w, a, o);
    } else {
      launch(k_conv2d<R>, out->size(), x.as<R>(), param(weight), b, cin, cout, x.h, x.w, k, a, o);
    }
    return out;
  }

  TensorPtr deform_conv2d(const TensorPtr& xp, const Param& offset_weight, const Param& offset_bias,
                          const Param& weight, const Param& bias, Act act) override {
    use();
    const CudaTensor& x = dev(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], h = x.h, w = x.w, hw = h * w;
    if (cin != x.c || weight.shape[2] != 3 || weight.shape[3] != 3 || offset_weight.shape[0] != 18)
      throw std::runtime_error("deform_conv2d: shape mismatch");
    TensorPtr offset = conv2d(xp, offset_weight, &offset_bias, Act::None);
    const float max_offset = static_cast<float>(std::max(h, w)) / 4.0f;
    Buffer<R> col(pool_, static_cast<size_t>(count(static_cast<long long>(cin) * 9 * hw)));
    g_profile.tag("deform sample %3d %4dx%-4d", cin, w, h);
    launch(k_deform_sample<R>, 9LL * hw, x.as<R>(), dev(offset).as<R>(), cin, h, w, max_offset, col.get());
    CudaPtr out = make(cout, h, w);
    const int t = tile(cout);
    const long long threads = static_cast<long long>((cout + t - 1) / t) * hw;
    const R *wt = param_tiled(weight, t), *b = param(bias);
    R* o = out->template out<R>();
    g_profile.tag("deform sum %3d->%-3d %4dx%-4d", cin, cout, w, h);
    if (t == 4) launch(k_deform_sum_tile<R, 4>, threads, col.get(), wt, b, cin * 9, cout, hw, act_id(act), o);
    else if (t == 8) launch(k_deform_sum_tile<R, 8>, threads, col.get(), wt, b, cin * 9, cout, hw, act_id(act), o);
    else launch(k_deform_sum_tile<R, 16>, threads, col.get(), wt, b, cin * 9, cout, hw, act_id(act), o);
    return out;
  }

  TensorPtr add(const TensorPtr& ap, const TensorPtr& bp, Act act) override {
    use();
    const CudaTensor &a = dev(ap), &b = dev(bp);
    if (a.size() != b.size()) throw std::runtime_error("add: shape mismatch");
    CudaPtr out = make(a.c, a.h, a.w);
    g_profile.tag("add c%d %dx%d", a.c, a.w, a.h);
    launch(k_add<R>, out->size(), a.as<R>(), b.as<R>(), out->size(), act_id(act), out->template out<R>());
    return out;
  }

  TensorPtr avg_pool(const TensorPtr& xp, int k) override {
    use();
    const CudaTensor& x = dev(xp);
    if (k < 1 || x.h % k || x.w % k) throw std::runtime_error("avg_pool: size is not a multiple of the window");
    CudaPtr out = make(x.c, x.h / k, x.w / k);
    g_profile.tag("avg_pool%d c%d %dx%d", k, x.c, x.w, x.h);
    launch(k_avg_pool<R>, out->size(), x.as<R>(), x.c, x.w, out->h, out->w, k, out->template out<R>());
    return out;
  }

  TensorPtr upsample(const TensorPtr& xp, int scale) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h * scale, x.w * scale);
    g_profile.tag("upsample%d c%d -> %dx%d", scale, x.c, out->w, out->h);
    launch(k_upsample<R>, out->size(), x.as<R>(), x.c, x.h, x.w, out->h, out->w, upsample_ratio(x.w, out->w),
           upsample_ratio(x.h, out->h), out->template out<R>());
    return out;
  }

  TensorPtr concat(const std::vector<TensorPtr>& xs) override {
    use();
    int c = 0;
    for (const auto& t : xs) {
      if (t->h != xs[0]->h || t->w != xs[0]->w) throw std::runtime_error("concat: shape mismatch");
      c += t->c;
    }
    CudaPtr out = make(c, xs[0]->h, xs[0]->w);
    size_t pos = 0;
    for (const auto& t : xs) {
      const CudaTensor& x = dev(t);
      if (x.size())
        JKP_CUDA_CHECK(cudaMemcpy(out->template out<R>() + pos, x.as<R>(), x.bytes(), cudaMemcpyDeviceToDevice));
      pos += x.size();
    }
    return out;
  }

  TensorPtr l2_normalize(const TensorPtr& xp) override { return l2_normalize_crop(xp, 0, 0, xp->w, xp->h); }

  // A pyramid: stored only when an operation needs its values (see the top of the file).
  TensorPtr upsample_concat(const std::vector<TensorPtr>& xs, const std::vector<int>& scales) override {
    if (xs.empty() || xs.size() != scales.size()) throw std::runtime_error("upsample_concat: bad arguments");
    if (xs.size() > 4) return Ops::upsample_concat(xs, scales);
    auto p = std::make_shared<Pyramid>();
    p->height = xs[0]->h * scales[0], p->width = xs[0]->w * scales[0];
    p->store = &CudaOps::store_pyramid;
    int c = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
      if (xs[i]->h * scales[i] != p->height || xs[i]->w * scales[i] != p->width)
        throw std::runtime_error("concat: shape mismatch");
      if (dev(xs[i]).half != kHalf) throw std::runtime_error("cuda: tensor has another element type");
      c += xs[i]->c;
      p->parts.push_back(std::static_pointer_cast<CudaTensor>(xs[i]));
      p->scales.push_back(scales[i]);
    }
    return std::make_shared<CudaTensor>(pool_, c, p->height, p->width, kHalf, p);
  }

  TensorPtr l2_normalize_crop(const TensorPtr& xp, int left, int top, int width, int height) override {
    use();
    const CudaTensor& x = dev(xp);
    if (x.pyramid && !x.pyramid->normalized) {
      auto p = std::make_shared<Pyramid>(*x.pyramid);
      p->normalized = true, p->left = left, p->top = top;
      return std::make_shared<CudaTensor>(pool_, x.c, height, width, kHalf, p);
    }
    CudaPtr out = make(x.c, height, width);
    g_profile.tag("l2_normalize_crop c%d %dx%d", x.c, width, height);
    launch(k_l2_normalize_crop<R>, width * height, x.as<R>(), x.c, x.h, x.w, left, top, height, width,
           out->template out<R>());
    return out;
  }

  void detect(const TensorPtr& score_map, const DetectParams& params, Keypoints* out) override {
    use();
    const CudaTensor& sm = dev(score_map);
    const int w = sm.w, h = sm.h, n = w * h;
    out->xy_norm.clear();
    out->score.clear();
    if (n == 0) return;
    const float* s = sm.as<float>();  // the score map is float in both modes

    // Non-maximum suppression (simple_nms from SuperGlue, two refinement rounds).
    Buffer<float> nms(pool_, n);
    g_profile.tag("detect nms %dx%d", w, h);
    {
      Buffer<float> tmp(pool_, n), mask(pool_, n), pooled_mask(pool_, n), supp(pool_, n), pooled_supp(pool_, n);
      auto max_filter = [&](const float* src, float* dst) {
        launch(k_max_filter, n, src, w, h, kRadius, 0, tmp.get());
        launch(k_max_filter, n, tmp.get(), w, h, kRadius, 1, dst);
      };
      max_filter(s, pooled_supp.get());
      launch(k_nms_init, n, s, pooled_supp.get(), n, mask.get());
      for (int round = 0; round < 2; ++round) {
        max_filter(mask.get(), pooled_mask.get());
        launch(k_nms_suppress, n, s, pooled_mask.get(), n, supp.get());
        max_filter(supp.get(), pooled_supp.get());
        launch(k_nms_update, n, supp.get(), pooled_supp.get(), pooled_mask.get(), n, mask.get());
      }
      launch(k_nms_final, n, s, mask.get(), w, h, kRadius, nms.get());
    }

    // Selection, on the device. `idx` receives the pixel indices of the keypoints in output order.
    g_profile.tag("detect select");
    Selection select(pool_, nms.get(), n);
    int kept = 0;         // number of keypoints
    int best = -1;        // >= 0: keep the `best` highest scores, best first
    float threshold = 0;  // otherwise: keep nms > threshold, in raster order
    if (params.top_k > 0) {
      best = std::min(params.top_k, n);
    } else {
      int passing = params.scores_th > 0 ? select.count(kGreater, threshold = params.scores_th) : 0;
      if (passing == 0) {  // fall back to the mean score
        Buffer<float> sum(pool_, 1);
        launch(k_sum_sequential, 1, s, n, sum.get());
        float total = 0;
        sum.download(&total, 1);
        passing = select.count(kGreater, threshold = total / static_cast<float>(n));
      }
      if (passing > params.n_limit) best = std::max(0, params.n_limit);
      else kept = passing;
    }
    if (best >= 0) kept = best;
    if (kept == 0) return;

    Buffer<int> idx(pool_, kept);
    if (best < 0) select.write(kGreater, threshold, kept, idx.get());
    else select.top(best, idx.get());

    // Refinement and readback of the kept keypoints only.
    Buffer<float> xy(pool_, 2 * static_cast<size_t>(kept)), score(pool_, kept);
    g_profile.tag("detect refine");
    launch(k_refine, kept, s, w, h, idx.get(), kept, xy.get(), score.get());
    out->xy_norm.resize(2 * static_cast<size_t>(kept));
    out->score.resize(kept);
    xy.download(out->xy_norm.data(), out->xy_norm.size());
    score.download(out->score.data(), out->score.size());
  }

  TensorPtr normalize(const TensorPtr& xp, const float* mean, const float* std) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h, x.w);
    std::vector<float> host(2 * static_cast<size_t>(x.c));
    for (int c = 0; c < x.c; ++c) host[c] = mean[c], host[x.c + c] = std[c];
    Buffer<float> ms(pool_, host.size());
    ms.upload(host.data(), host.size());
    g_profile.tag("normalize c%d %dx%d", x.c, x.w, x.h);
    launch(k_normalize<R>, out->size(), x.as<R>(), x.c, x.h * x.w, ms.get(), out->template out<R>());
    return out;
  }

  TensorPtr pad_reflect(const TensorPtr& xp, int p) override {
    use();
    const CudaTensor& x = dev(xp);
    if (p < 0 || p >= x.h || p >= x.w) throw std::runtime_error("pad_reflect: padding does not fit the tensor");
    CudaPtr out = make(x.c, x.h + 2 * p, x.w + 2 * p);
    g_profile.tag("pad_reflect c%d %dx%d", x.c, out->w, out->h);
    launch(k_pad_reflect<R>, out->size(), x.as<R>(), x.c, x.h, x.w, p, out->template out<R>());
    return out;
  }

  void detect_raco(const TensorPtr& logits_p, const TensorPtr& ranker_map, const TensorPtr& cholesky_raw,
                   const RacoDetectParams& params, RacoKeypoints* out) override {
    use();
    const CudaTensor& lm = dev(logits_p);
    const int w = lm.w, h = lm.h, n = w * h, k = params.window, r = k / 2;
    if (lm.c != 1 || k < 1 || k % 2 == 0 || k > 9) throw std::runtime_error("detect_raco: bad arguments");
    if ((ranker_map && ranker_map->c != 1) || (cholesky_raw && cholesky_raw->c != 3))
      throw std::runtime_error("detect_raco: bad arguments");
    for (const TensorPtr& t : {ranker_map, cholesky_raw})
      if (t && (t->w != w || t->h != h)) throw std::runtime_error("detect_raco: map size mismatch");
    const R* logits = lm.as<R>();
    const R* ranker = ranker_map ? dev(ranker_map).as<R>() : nullptr;
    const R* chol = cholesky_raw ? dev(cholesky_raw).as<R>() : nullptr;
    out->xy.clear(), out->score.clear(), out->rank.clear(), out->cholesky.clear();
    if (n == 0) return;

    // Probabilities: softmax over the whole image (stats: maximum logit, sum of the exponentials).
    g_profile.tag("detect_raco softmax %dx%d", w, h);
    const int blocks = (n + kScanBlock - 1) / kScanBlock;
    Buffer<float> partial_max(pool_, blocks), stats(pool_, 2), prob(pool_, n), nms(pool_, n);
    Buffer<double> partial_sum(pool_, blocks);
    launch(k_softmax_block_max<R>, blocks, logits, n, partial_max.get());
    launch(k_softmax_max, 1, partial_max.get(), blocks, stats.get());
    launch(k_softmax_block_sum<R>, blocks, logits, n, stats.get(), partial_sum.get());
    launch(k_softmax_sum, 1, partial_sum.get(), blocks, stats.get());
    launch(k_softmax_prob<R>, n, logits, n, stats.get(), prob.get());

    // Non-maximum suppression: keep pixels equal to the maximum of their window.
    g_profile.tag("detect_raco nms %dx%d", w, h);
    {
      Buffer<float> tmp(pool_, n), pooled(pool_, n);
      launch(k_max_filter, n, prob.get(), w, h, r, 0, tmp.get());
      launch(k_max_filter, n, tmp.get(), w, h, r, 1, pooled.get());
      launch(k_raco_nms, n, prob.get(), pooled.get(), n, nms.get());
    }

    // The best top_k of the maxima; suppressed pixels (zero) are not keypoints.
    g_profile.tag("detect_raco select");
    Selection select(pool_, nms.get(), n);
    const int kept = std::min(std::max(params.top_k, 0), select.count(kGreater, 0.0f));
    if (kept == 0) return;
    Buffer<int> idx(pool_, kept);
    select.top(kept, idx.get());

    // Refinement and sampling of the kept keypoints only.
    const size_t nk = static_cast<size_t>(kept);
    Buffer<float> xy(pool_, 2 * nk), score(pool_, nk), rank(pool_, ranker ? nk : 0), cov(pool_, chol ? 3 * nk : 0);
    g_profile.tag("detect_raco refine");
    launch(k_raco_refine<R>, kept, logits, prob.get(), ranker, chol, w, h, idx.get(), kept, k, params.temperature,
           xy.get(), score.get(), rank.get(), cov.get());
    out->xy.resize(2 * nk);
    out->score.resize(nk);
    xy.download(out->xy.data(), out->xy.size());
    score.download(out->score.data(), out->score.size());
    if (ranker) {
      out->rank.resize(nk);
      rank.download(out->rank.data(), nk);
    }
    if (chol) {
      out->cholesky.resize(3 * nk);
      cov.download(out->cholesky.data(), 3 * nk);
    }
  }

  void describe(const TensorPtr& feature_map, const Keypoints& kpts, const SddhParams& p,
                std::vector<float>* desc) override {
    use();
    const CudaTensor& f = dev(feature_map);
    const int c = f.c, h = f.h, w = f.w, total = kpts.n(), K = p.kernel, M = p.positions;
    const int dim = p.agg_w->shape[2];
    if (p.offset0_w->shape[1] != c || p.agg_w->shape[1] != c || p.sf_w->shape[0] != c)
      throw std::runtime_error("describe: shape mismatch");
    const float max_offset = static_cast<float>(std::max(h, w)) / 4.0f;
    desc->assign(static_cast<size_t>(total) * dim, 0.0f);
    // On a normalized pyramid the head normalizes only the pixels it reads.
    const bool lazy = f.pyramid && f.pyramid->normalized && c <= kMaxPyramidC;
    const PyramidArgs args = lazy ? f.pyramid_args() : PyramidArgs();
    const R *p0 = lazy ? part(f, 0) : nullptr, *p1 = lazy ? part(f, 1) : nullptr, *p2 = lazy ? part(f, 2) : nullptr,
            *p3 = lazy ? part(f, 3) : nullptr;
    const R* stored = lazy ? nullptr : f.as<R>();
    const float *off0_w = param_f32(*p.offset0_w), *off0_b = param_f32(*p.offset0_b), *off1_w = param_f32(*p.offset1_w),
                *off1_b = param_f32(*p.offset1_b), *sf_w = param_tiled_f32(*p.sf_w, 1), *agg_w = param_f32(*p.agg_w);

    // In chunks, so the per-keypoint work buffers (M*C floats each) stay small for any keypoint count.
    const int chunk = 8192;
    for (int begin = 0; begin < total; begin += chunk) {
      const int n = std::min(chunk, total - begin);
      const size_t samples = static_cast<size_t>(n) * M * c;
      Buffer<float> xy(pool_, 2 * static_cast<size_t>(n)), o0(pool_, static_cast<size_t>(n) * 2 * M),
          o1(pool_, static_cast<size_t>(n) * 2 * M), sampled(pool_, samples), feat(pool_, samples),
          d(pool_, static_cast<size_t>(n) * dim);
      xy.upload(kpts.xy_norm.data() + 2 * static_cast<size_t>(begin), xy.size());
      g_profile.tag("describe offsets");
      if (lazy) {
        Buffer<float> patch(pool_, static_cast<size_t>(n) * K * K * c);
        g_profile.tag("describe patch");
        launch(k_sddh_patch_pyramid<R>, n * K * K, p0, p1, p2, p3, args, h, w, xy.get(), n, K, patch.get());
        g_profile.tag("describe offsets");
        launch(k_sddh_offset0_patch, n * 2 * M, patch.get(), c, h, w, xy.get(), n, K, off0_w, off0_b, 2 * M, o0.get());
      } else {
        launch(k_sddh_offset0<R>, n * 2 * M, stored, c, h, w, xy.get(), n, K, off0_w, off0_b, 2 * M, o0.get());
      }
      launch(k_sddh_offset1, n * 2 * M, o0.get(), n, 2 * M, off1_w, off1_b, max_offset, o1.get());
      g_profile.tag("describe sample");
      if (lazy)
        launch(k_sddh_sample_pyramid<R>, n * M, p0, p1, p2, p3, args, h, w, xy.get(), o1.get(), n, M, sampled.get());
      else
        launch(k_sddh_sample<R>, n * M, stored, c, h, w, xy.get(), o1.get(), n, M, sampled.get());
      g_profile.tag("describe sf");
      // One output per thread for these two: on CUDA more, smaller threads beat tiling here (measured).
      launch(k_sddh_sf_tile<R, 1>, static_cast<long long>(n) * M * c, sampled.get(), count(samples), c, sf_w, feat.get());
      g_profile.tag("describe aggregate");
      launch(k_sddh_aggregate_tile<R, 1>, static_cast<long long>(n) * dim, feat.get(), n, M * c, dim, agg_w, d.get());
      g_profile.tag("describe normalize");
      launch(k_normalize_rows, n, d.get(), n, dim);
      d.download(desc->data() + static_cast<size_t>(begin) * dim, d.size());
    }
  }

 private:
  // The current device is per host thread; select ours on entry to every operation.
  void use() const { JKP_CUDA_CHECK(cudaSetDevice(pool_->device())); }

  // A tensor in the storage type.
  CudaPtr make(int c, int h, int w) { return std::make_shared<CudaTensor>(pool_, c, h, w, kHalf); }

  // Output channels per thread for the tiled kernels (instantiated for 4, 8 and 16).
  static int tile(int cout) {
    static const int forced = [] {
      const char* env = std::getenv("JKP_CUDA_TILE");  // tuning aid
      return env ? std::atoi(env) : 0;
    }();
    if (forced == 4 || forced == 8 || forced == 16) return forced;
    return cout <= 4 ? 4 : cout <= 8 ? 8 : 16;
  }

  // Part i of a pyramid tensor, or part 0 for an unused slot (its channel count is 0).
  static const R* part(const CudaTensor& x, size_t i) {
    const auto& parts = x.pyramid->parts;
    return parts[i < parts.size() ? i : 0]->template as<R>();
  }

  // Computes and stores the values of a pyramid tensor (the eager form of upsample_concat and
  // l2_normalize_crop).
  static void store_pyramid(const CudaTensor& t, void* dst) {
    const Pyramid& p = *t.pyramid;
    int channels = 0;
    for (const auto& x : p.parts) channels += x->c;
    const size_t plane = static_cast<size_t>(p.height) * p.width;
    std::unique_ptr<Buffer<R>> full;  // the concatenation, when it is not the result itself
    R* cat = static_cast<R*>(dst);
    if (p.normalized) {
      full = std::make_unique<Buffer<R>>(t.pool, channels * plane);
      cat = full->get();
    }
    size_t pos = 0;
    for (size_t i = 0; i < p.parts.size(); ++i) {
      const CudaTensor& x = *p.parts[i];
      const size_t values = x.c * plane;
      if (p.scales[i] == 1) {
        if (values)
          JKP_CUDA_CHECK(cudaMemcpy(cat + pos, x.template as<R>(), values * sizeof(R), cudaMemcpyDeviceToDevice));
      } else {
        g_profile.tag("upsample%d c%d -> %dx%d", p.scales[i], x.c, p.width, p.height);
        launch(k_upsample<R>, count(static_cast<long long>(values)), x.template as<R>(), x.c, x.h, x.w, p.height, p.width,
               upsample_ratio(x.w, p.width), upsample_ratio(x.h, p.height), cat + pos);
      }
      pos += values;
    }
    g_profile.tag("l2_normalize_crop c%d %dx%d", t.c, t.w, t.h);
    if (p.normalized)
      launch(k_l2_normalize_crop<R>, t.w * t.h, cat, t.c, p.height, p.width, p.left, p.top, t.h, t.w, static_cast<R*>(dst));
    // `full` returns to the pool here; the kernel that reads it was launched before any later one can reuse it.
  }

  // Device copies of weight tensors, uploaded on first use. Weights outlive the backend (see jkp_ctx).
  template <typename T>
  const T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    JKP_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(host.size(), 1) * sizeof(T)));
    params_owned_.push_back(d);
    if (!host.empty()) JKP_CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
  }
  template <typename T>
  const T* cached(std::map<std::pair<const float*, int>, const void*>& cache, const Param& p, int tile) {
    const auto key = std::make_pair(p.data, tile);
    auto it = cache.find(key);
    if (it != cache.end()) return static_cast<const T*>(it->second);
    std::vector<T> host;
    if (tile == 0) {  // as stored: [cout][...]
      host.resize(p.size());
      for (size_t i = 0; i < p.size(); ++i) encode(p.data[i], &host[i]);
    } else {  // tap-major, cout padded with zeros to a multiple of `tile` (see k_conv2d_tile)
      const size_t cout = static_cast<size_t>(p.shape[0]), taps = cout ? p.size() / cout : 0;
      const size_t cpad = (cout + tile - 1) / tile * tile;
      host.resize(taps * cpad);
      for (size_t i = 0; i < host.size(); ++i) encode(0.0f, &host[i]);
      for (size_t oc = 0; oc < cout; ++oc)
        for (size_t t = 0; t < taps; ++t) encode(p.data[oc * taps + t], &host[t * cpad + oc]);
    }
    const T* d = upload(host);
    cache[key] = d;
    return d;
  }
  // In the storage type / always float (descriptor head).
  const R* param(const Param& p) { return cached<R>(params_, p, 0); }
  const R* param_tiled(const Param& p, int tile) { return cached<R>(params_, p, tile); }
  const float* param_f32(const Param& p) { return cached<float>(params_f32_, p, 0); }
  const float* param_tiled_f32(const Param& p, int tile) { return cached<float>(params_f32_, p, tile); }

  PoolPtr pool_;
  std::map<std::pair<const float*, int>, const void*> params_, params_f32_;
  std::vector<void*> params_owned_;
};

}  // namespace

std::unique_ptr<Ops> make_cuda_ops(int device, Precision precision) {
  if (precision == Precision::Fp16) return std::make_unique<CudaOps<__half>>(device);
  return std::make_unique<CudaOps<float>>(device);
}

}  // namespace jkp
