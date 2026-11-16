// Metal backend kernels: one compute kernel per operation, FP32, CHW layout, same graph as the CPU backend.
//
// Each kernel is a port of the CUDA kernel of the same name (src/backend/cuda/cuda_ops.cu) and keeps the
// float32 operation order of src/backend/cpu/cpu_ops.cpp. Contraction into fused multiply-add is off and
// the library is compiled in safe math mode with precise functions (see metal_ops.mm), so the backends
// differ only through exp() (SELU, sigmoid, soft-argmax).
//
// The source is compiled at run time (newLibraryWithSource), so building needs no Metal toolchain.
// Scalar arguments are passed one per buffer index with setBytes, in the order of the CUDA signatures.
//
// Precision: `real` is the storage type of feature maps and of the weights of the dense network. It is
// float by default and half when the source is compiled with JKP_FP16. Every value is converted to float
// when it is read, so the arithmetic is float32 in both modes and the FP32 results do not depend on this.
// The score map (the sigmoid output), keypoint selection and the descriptor head's own buffers and
// weights are float in both modes.
#include <metal_stdlib>
using namespace metal;

#pragma METAL fp contract(off)

#ifdef JKP_FP16
typedef half real;
typedef packed_half4 real4;
// acc += w * v. FP16 mode is not tied to the CPU backend's rounding, so it uses a fused multiply-add.
#define JKP_MAC(acc, w, v) acc = fma(w, v, acc)
#else
typedef float real;
typedef packed_float4 real4;
#define JKP_MAC(acc, w, v) acc += (w) * (v)
#endif

// Weights of the tiled kernels are stored tap-major: w[(tap * cpad + oc)], tap = (ic * K + ky) * K + kx,
// cpad = cout rounded up to the tile size T (zeros in the padding). The T weights one thread needs for a
// tap are then adjacent and are read four at a time.

constant int kActNone = 0;
constant int kActSelu = 1;
constant int kActSigmoid = 2;
constant int kActLeakyRelu = 3;  // slope 0.01
constant int kGreater = 0;
constant int kRadius = 2;       // NMS and soft-argmax window radius
constant int kScanBlock = 256;  // elements handled by one thread in the compaction kernels

// ---------------------------------------------------------------------------------------------
// Helpers shared by the kernels
// ---------------------------------------------------------------------------------------------

inline float selu(float x) {
  const float alpha = 1.6732632423543772848170429916717f, scale = 1.0507009873554804934193349852946f;
  return x > 0 ? scale * x : scale * (alpha * (precise::exp(x) - 1.0f));
}

inline float activate(float v, int act) {
  if (act == kActSelu) return selu(v);
  if (act == kActSigmoid) return 1.0f / (1.0f + precise::exp(-v));
  if (act == kActLeakyRelu) return v > 0 ? v : 0.01f * v;
  return v;
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (hi < v ? hi : v); }

// Bilinear sample with zeros outside the map (PyTorch grid_sample, padding_mode = zeros).
inline void bilinear_setup(float x, float y, int w, int h, thread int* idx, thread float* wt) {
  float x0f = precise::floor(x), y0f = precise::floor(y);
  int x0 = int(x0f), y0 = int(y0f), x1 = x0 + 1, y1 = y0 + 1;
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
inline float bilinear_apply(device const P* plane, thread const int* idx, thread const float* wt) {
  return wt[0] * float(plane[idx[0]]) + wt[1] * float(plane[idx[1]]) + wt[2] * float(plane[idx[2]]) +
         wt[3] * float(plane[idx[3]]);
}

// ---------------------------------------------------------------------------------------------
// Dense kernels. One thread per output element unless noted.
// ---------------------------------------------------------------------------------------------

kernel void k_image(device const uchar* src [[buffer(0)]], constant int& stride [[buffer(1)]],
                    constant int& channels [[buffer(2)]], constant int& w [[buffer(3)]], constant int& h [[buffer(4)]],
                    device real* dst [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), hw = h * w;
  if (i >= 3 * hw) return;
  int c = i / hw, y = (i % hw) / w, x = i % w;
  int src_c = channels == 1 ? 0 : c;
  dst[i] = real(float(src[size_t(y) * stride + x * channels + src_c]) / 255.0f);
}

kernel void k_pad_replicate(device const real* src [[buffer(0)]], constant int& c [[buffer(1)]],
                            constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                            constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                            constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]],
                            device real* dst [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  int sy = min(max(y - top, 0), h - 1), sx = min(max(x - left, 0), w - 1);
  dst[i] = src[(size_t(ch) * h + sy) * w + sx];
}

template <typename P>
kernel void k_crop_any(device const P* src [[buffer(0)]], constant int& c [[buffer(1)]], constant int& h [[buffer(2)]],
                       constant int& w [[buffer(3)]], constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                       constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]], device P* dst [[buffer(8)]],
                       uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  dst[i] = src[(size_t(ch) * h + y + top) * w + x + left];
}
#define JKP_CROP(NAME, P)                                                                                   \
  template [[host_name(NAME)]] kernel void k_crop_any<P>(device const P*, constant int&, constant int&,     \
      constant int&, constant int&, constant int&, constant int&, constant int&, device P*, uint);
JKP_CROP("k_crop", real)
#ifdef JKP_FP16
JKP_CROP("k_crop_f32", float)  // the score map
#endif

kernel void k_conv2d(device const real* x [[buffer(0)]], device const real* weight [[buffer(1)]],
                     device const real* bias [[buffer(2)]], constant int& has_bias [[buffer(3)]],
                     constant int& cin [[buffer(4)]], constant int& cout [[buffer(5)]], constant int& h [[buffer(6)]],
                     constant int& w [[buffer(7)]], constant int& k [[buffer(8)]], constant int& act [[buffer(9)]],
                     device real* out [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), hw = h * w;
  if (i >= cout * hw) return;
  int oc = i / hw, y = (i % hw) / w, xo = i % w, pad = k / 2;
  float acc = has_bias ? float(bias[oc]) : 0.0f;
  device const real* wk = weight + size_t(oc) * cin * k * k;
  for (int ic = 0; ic < cin; ++ic, wk += k * k) {
    device const real* in = x + size_t(ic) * hw;
    for (int ky = 0; ky < k; ++ky) {
      int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < k; ++kx) {
        int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        acc += float(wk[ky * k + kx]) * float(in[sy * w + sx]);
      }
    }
  }
  out[i] = real(activate(acc, act));
}

// Convolution, T output channels per thread for one pixel, KxK kernel known at compile time. Each input
// value is read once for all T outputs instead of once per output, and each output accumulates in the
// same order as k_conv2d, so the results are identical. Weights are tap-major (see above), uniform across
// the threads of a SIMD group and read through the constant address space.
// Grid: ceil(cout / T) * h * w, pixel fastest. T is a multiple of 4.
// O is the output storage type: `real`, or float for the score map in FP16 mode.
template <int T, int K, typename O>
kernel void k_conv2d_tile(device const real* x [[buffer(0)]], constant real4* weight [[buffer(1)]],
                          constant real* bias [[buffer(2)]], constant int& has_bias [[buffer(3)]],
                          constant int& cin [[buffer(4)]], constant int& cout [[buffer(5)]], constant int& h [[buffer(6)]],
                          constant int& w [[buffer(7)]], constant int& act [[buffer(8)]], device O* out [[buffer(9)]],
                          uint gid [[thread_position_in_grid]]) {
  const int hw = h * w, blocks = (cout + T - 1) / T;
  const int i = int(gid);
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, y = pix / w, xo = pix % w, pad = K / 2, oc0 = ob * T;
  float4 acc[T / 4];
  for (int t = 0; t < T; ++t) acc[t / 4][t % 4] = has_bias && oc0 + t < cout ? float(bias[oc0 + t]) : 0.0f;
  for (int ic = 0; ic < cin; ++ic) {
    device const real* in = x + size_t(ic) * hw;
    const int wic = ic * K * K;
    for (int ky = 0; ky < K; ++ky) {
      const int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < K; ++kx) {
        const int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        const float4 v = float4(float(in[sy * w + sx]));
        constant real4* wp = weight + ((wic + ky * K + kx) * blocks + ob) * (T / 4);
        for (int q = 0; q < T / 4; ++q) JKP_MAC(acc[q], float4(wp[q]), v);
      }
    }
  }
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) out[size_t(oc0 + t) * hw + pix] = O(activate(acc[t / 4][t % 4], act));
}

#define JKP_CONV_TILE(NAME, T, K, O)                                                                     \
  template [[host_name(NAME)]] kernel void k_conv2d_tile<T, K, O>(                                        \
      device const real*, constant real4*, constant real*, constant int&, constant int&, constant int&,  \
      constant int&, constant int&, constant int&, device O*, uint);
JKP_CONV_TILE("k_conv2d_t4_k1", 4, 1, real)
JKP_CONV_TILE("k_conv2d_t8_k1", 8, 1, real)
JKP_CONV_TILE("k_conv2d_t16_k1", 16, 1, real)
JKP_CONV_TILE("k_conv2d_t32_k1", 32, 1, real)
JKP_CONV_TILE("k_conv2d_t4_k3", 4, 3, real)
JKP_CONV_TILE("k_conv2d_t8_k3", 8, 3, real)
JKP_CONV_TILE("k_conv2d_t16_k3", 16, 3, real)
JKP_CONV_TILE("k_conv2d_t32_k3", 32, 3, real)
#ifdef JKP_FP16
JKP_CONV_TILE("k_conv2d_t4_k1_f32", 4, 1, float)  // the score map
JKP_CONV_TILE("k_conv2d_t4_k3_f32", 4, 3, float)
#endif

// Deformable convolution, step 1. One thread per (kernel tap, pixel): samples every input channel at
// the offset position. col[(ic * 9 + tap) * hw + pixel].
kernel void k_deform_sample(device const real* x [[buffer(0)]], device const real* off [[buffer(1)]],
                            constant int& cin [[buffer(2)]], constant int& h [[buffer(3)]], constant int& w [[buffer(4)]],
                            constant float& max_offset [[buffer(5)]], device real* col [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
  int i = int(gid), hw = h * w;
  if (i >= 9 * hw) return;
  int k = i / hw, pix = i % hw, y = pix / w, xo = pix % w;
  float off_y = off[size_t(2 * k) * hw + pix], off_x = off[size_t(2 * k + 1) * hw + pix];
  float py = float(y - 1 + k / 3) + clampf(off_y, -max_offset, max_offset);
  float px = float(xo - 1 + k % 3) + clampf(off_x, -max_offset, max_offset);
  int idx[4] = {0, 0, 0, 0};
  float wt[4] = {0, 0, 0, 0};
  if (py > -1 && py < h && px > -1 && px < w) bilinear_setup(px, py, w, h, idx, wt);
  for (int ic = 0; ic < cin; ++ic)
    col[(size_t(ic) * 9 + k) * hw + pix] = real(bilinear_apply(x + size_t(ic) * hw, idx, wt));
}

// Deformable convolution, step 2: weighted sum over the sampled columns.
kernel void k_deform_sum(device const real* col [[buffer(0)]], device const real* weight [[buffer(1)]],
                         device const real* bias [[buffer(2)]], constant int& taps [[buffer(3)]],
                         constant int& cout [[buffer(4)]], constant int& hw [[buffer(5)]], constant int& act [[buffer(6)]],
                         device real* out [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= cout * hw) return;
  int oc = i / hw, pix = i % hw;
  device const real* wk = weight + size_t(oc) * taps;
  float acc = float(bias[oc]);
  for (int j = 0; j < taps; ++j) acc += float(wk[j]) * float(col[size_t(j) * hw + pix]);
  out[i] = real(activate(acc, act));
}

// Deformable convolution, step 2, T output channels per thread (same order per output as k_deform_sum).
// Weights are tap-major, tap = ic * 9 + kernel tap.
template <int T>
kernel void k_deform_sum_tile(device const real* col [[buffer(0)]], constant real4* weight [[buffer(1)]],
                              constant real* bias [[buffer(2)]], constant int& taps [[buffer(3)]],
                              constant int& cout [[buffer(4)]], constant int& hw [[buffer(5)]],
                              constant int& act [[buffer(6)]], device real* out [[buffer(7)]],
                              uint gid [[thread_position_in_grid]]) {
  const int blocks = (cout + T - 1) / T;
  const int i = int(gid);
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, oc0 = ob * T;
  float4 acc[T / 4];
  for (int t = 0; t < T; ++t) acc[t / 4][t % 4] = oc0 + t < cout ? float(bias[oc0 + t]) : 0.0f;
  for (int j = 0; j < taps; ++j) {
    const float4 v = float4(float(col[size_t(j) * hw + pix]));
    constant real4* wp = weight + (j * blocks + ob) * (T / 4);
    for (int q = 0; q < T / 4; ++q) JKP_MAC(acc[q], float4(wp[q]), v);
  }
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) out[size_t(oc0 + t) * hw + pix] = real(activate(acc[t / 4][t % 4], act));
}

#define JKP_DEFORM_TILE(T)                                                                                   \
  template [[host_name("k_deform_sum_t" #T)]] kernel void k_deform_sum_tile<T>(                              \
      device const real*, constant real4*, constant real*, constant int&, constant int&, constant int&,     \
      constant int&, device real*, uint);
JKP_DEFORM_TILE(4)
JKP_DEFORM_TILE(8)
JKP_DEFORM_TILE(16)
JKP_DEFORM_TILE(32)

kernel void k_add(device const real* a [[buffer(0)]], device const real* b [[buffer(1)]], constant int& n [[buffer(2)]],
                  constant int& act [[buffer(3)]], device real* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  out[i] = real(activate(float(a[i]) + float(b[i]), act));
}

kernel void k_avg_pool(device const real* x [[buffer(0)]], constant int& c [[buffer(1)]], constant int& w [[buffer(2)]],
                       constant int& oh [[buffer(3)]], constant int& ow [[buffer(4)]], constant int& k [[buffer(5)]],
                       device real* out [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  device const real* in = x + size_t(ch) * (oh * k) * w;
  float sum = 0;
  for (int j = 0; j < k; ++j)
    for (int q = 0; q < k; ++q) sum += float(in[(y * k + j) * w + xo * k + q]);
  out[i] = real(sum / float(k * k));
}

// Bilinear, align_corners = true: source position = dst * (in - 1) / (out - 1); sx, sy are those ratios.
kernel void k_upsample(device const real* x [[buffer(0)]], constant int& c [[buffer(1)]], constant int& h [[buffer(2)]],
                       constant int& w [[buffer(3)]], constant int& oh [[buffer(4)]], constant int& ow [[buffer(5)]],
                       constant float& sx [[buffer(6)]], constant float& sy [[buffer(7)]], device real* out [[buffer(8)]],
                       uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  float src_x = sx * float(xo), src_y = sy * float(y);
  int x0 = min(int(src_x), w - 1), y0 = min(int(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  float w1 = src_x - float(x0), w0 = 1.0f - w1;
  float h1 = src_y - float(y0), h0 = 1.0f - h1;
  device const real* r0 = x + (size_t(ch) * h + y0) * w;
  device const real* r1 = x + (size_t(ch) * h + y1) * w;
  out[i] = real(h0 * (w0 * float(r0[x0]) + w1 * float(r0[x1])) + h1 * (w0 * float(r1[x0]) + w1 * float(r1[x1])));
}

kernel void k_copy(device const real* src [[buffer(0)]], constant int& n [[buffer(1)]], device real* dst [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  dst[i] = src[i];
}

// One thread per pixel.
kernel void k_l2_normalize(device const real* x [[buffer(0)]], constant int& c [[buffer(1)]],
                           constant int& hw [[buffer(2)]], device real* out [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= hw) return;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = float(x[size_t(ch) * hw + i]);
    norm += v * v;
  }
  norm = fmax(precise::sqrt(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch) out[size_t(ch) * hw + i] = real(float(x[size_t(ch) * hw + i]) / norm);
}

// L2 normalization across channels of the cropped region of x (c x h x w) at (left, top); out is
// c x oh x ow. Same arithmetic as k_l2_normalize followed by k_crop. One thread per output pixel.
kernel void k_l2_normalize_crop(device const real* x [[buffer(0)]], constant int& c [[buffer(1)]],
                                constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                                constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                                constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]],
                                device real* out [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  const int i = int(gid), ohw = oh * ow;
  if (i >= ohw) return;
  const int hw = h * w, src = (i / ow + top) * w + i % ow + left;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = float(x[size_t(ch) * hw + src]);
    norm += v * v;
  }
  norm = fmax(precise::sqrt(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch) out[size_t(ch) * ohw + i] = real(float(x[size_t(ch) * hw + src]) / norm);
}

// ---------------------------------------------------------------------------------------------
// Pyramid: the concatenation of up to four maps, each upsampled to the full resolution (k_upsample's
// arithmetic, or as is for scale 1). The kernels below compute its values where they are needed
// instead of storing it: 128 channels at full resolution is the largest tensor of the network.
// In FP32 they give exactly what the stored tensor would hold.
// ---------------------------------------------------------------------------------------------

struct PyramidArgs {
  int c[4], h[4], w[4], scale[4];  // per part; c = 0 for an unused part
  float sx[4], sy[4];              // k_upsample's ratios
  int H, W;                        // full resolution
  int left, top;                   // origin of the cropped map inside the pyramid (descriptor kernels)
};

// Where one part is read for a full-resolution pixel.
struct Taps {
  int i00, i01, i10, i11;
  float w0, w1, h0, h1;
  bool direct;
};

inline Taps part_taps(constant PyramidArgs& a, int p, int X, int Y) {
  Taps t;
  const int w = a.w[p], h = a.h[p];
  t.direct = a.scale[p] == 1;
  if (t.direct) {
    t.i00 = t.i01 = t.i10 = t.i11 = Y * w + X;
    t.w0 = t.h0 = 1.0f, t.w1 = t.h1 = 0.0f;
    return t;
  }
  float src_x = a.sx[p] * float(X), src_y = a.sy[p] * float(Y);
  int x0 = min(int(src_x), w - 1), y0 = min(int(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  t.w1 = src_x - float(x0), t.w0 = 1.0f - t.w1;
  t.h1 = src_y - float(y0), t.h0 = 1.0f - t.h1;
  t.i00 = y0 * w + x0, t.i01 = y0 * w + x1, t.i10 = y1 * w + x0, t.i11 = y1 * w + x1;
  return t;
}

// Channel `ch` of part p (stored at m) at those taps.
inline float part_value(device const real* m, constant PyramidArgs& a, int p, int ch, Taps t) {
  device const real* plane = m + size_t(ch) * (a.h[p] * a.w[p]);
  if (t.direct) return float(plane[t.i00]);
  return t.h0 * (t.w0 * float(plane[t.i00]) + t.w1 * float(plane[t.i01])) +
         t.h1 * (t.w0 * float(plane[t.i10]) + t.w1 * float(plane[t.i11]));
}

// 1x1 convolution on a pyramid, T output channels per thread; same accumulation order as
// k_conv2d_tile<T, 1> on the stored pyramid, tap-major weights. Grid: ceil(cout / T) * H * W, pixel fastest.
template <int T>
inline void pyramid_accumulate(device const real* m, constant PyramidArgs& a, int p, int X, int Y,
                               constant real4* weight, int blocks, int ob, thread int& base, thread float4* acc) {
  if (a.c[p] == 0) return;
  const Taps taps = part_taps(a, p, X, Y);
  for (int ch = 0; ch < a.c[p]; ++ch) {
    const float4 v = float4(part_value(m, a, p, ch, taps));
    constant real4* wp = weight + ((base + ch) * blocks + ob) * (T / 4);
    for (int q = 0; q < T / 4; ++q) JKP_MAC(acc[q], float4(wp[q]), v);
  }
  base += a.c[p];
}

template <int T, typename O>
kernel void k_conv1x1_pyramid(device const real* p0 [[buffer(0)]], device const real* p1 [[buffer(1)]],
                              device const real* p2 [[buffer(2)]], device const real* p3 [[buffer(3)]],
                              constant PyramidArgs& a [[buffer(4)]], constant real4* weight [[buffer(5)]],
                              constant real* bias [[buffer(6)]], constant int& has_bias [[buffer(7)]],
                              constant int& cout [[buffer(8)]], constant int& act [[buffer(9)]],
                              device O* out [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  const int hw = a.H * a.W, blocks = (cout + T - 1) / T;
  const int i = int(gid);
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, Y = pix / a.W, X = pix % a.W, oc0 = ob * T;
  float4 acc[T / 4];
  for (int t = 0; t < T; ++t) acc[t / 4][t % 4] = has_bias && oc0 + t < cout ? float(bias[oc0 + t]) : 0.0f;
  int base = 0;
  pyramid_accumulate<T>(p0, a, 0, X, Y, weight, blocks, ob, base, acc);
  pyramid_accumulate<T>(p1, a, 1, X, Y, weight, blocks, ob, base, acc);
  pyramid_accumulate<T>(p2, a, 2, X, Y, weight, blocks, ob, base, acc);
  pyramid_accumulate<T>(p3, a, 3, X, Y, weight, blocks, ob, base, acc);
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) out[size_t(oc0 + t) * hw + pix] = O(activate(acc[t / 4][t % 4], act));
}

#define JKP_CONV_PYRAMID(NAME, T, O)                                                                        \
  template [[host_name(NAME)]] kernel void k_conv1x1_pyramid<T, O>(                                          \
      device const real*, device const real*, device const real*, device const real*, constant PyramidArgs&, \
      constant real4*, constant real*, constant int&, constant int&, constant int&, device O*, uint);
JKP_CONV_PYRAMID("k_conv1x1_pyramid_t4", 4, real)
JKP_CONV_PYRAMID("k_conv1x1_pyramid_t8", 8, real)
JKP_CONV_PYRAMID("k_conv1x1_pyramid_t16", 16, real)

// L2 norm across the channels of the pyramid at one pixel, as k_l2_normalize computes it.
inline float pyramid_norm(device const real* p0, device const real* p1, device const real* p2, device const real* p3,
                          constant PyramidArgs& a, thread const Taps* taps) {
  float norm = 0;
  for (int ch = 0; ch < a.c[0]; ++ch) { float v = part_value(p0, a, 0, ch, taps[0]); norm += v * v; }
  for (int ch = 0; ch < a.c[1]; ++ch) { float v = part_value(p1, a, 1, ch, taps[1]); norm += v * v; }
  for (int ch = 0; ch < a.c[2]; ++ch) { float v = part_value(p2, a, 2, ch, taps[2]); norm += v * v; }
  for (int ch = 0; ch < a.c[3]; ++ch) { float v = part_value(p3, a, 3, ch, taps[3]); norm += v * v; }
  return fmax(precise::sqrt(norm), 1e-12f);
}

// ---------------------------------------------------------------------------------------------
// Detection kernels
// ---------------------------------------------------------------------------------------------

// Separable (2r+1) max filter, window clipped at the borders. dir 0 = along x, 1 = along y.
kernel void k_max_filter(device const float* src [[buffer(0)]], constant int& w [[buffer(1)]],
                         constant int& h [[buffer(2)]], constant int& r [[buffer(3)]], constant int& dir [[buffer(4)]],
                         device float* dst [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= w * h) return;
  int y = i / w, x = i % w;
  float m = src[i];
  if (dir == 0) {
    for (int k = max(0, x - r); k <= min(w - 1, x + r); ++k) m = fmax(m, src[y * w + k]);
  } else {
    for (int k = max(0, y - r); k <= min(h - 1, y + r); ++k) m = fmax(m, src[k * w + x]);
  }
  dst[i] = m;
}

// mask = 1 where the score is the maximum of its window.
kernel void k_nms_init(device const float* s [[buffer(0)]], device const float* pooled [[buffer(1)]],
                       constant int& n [[buffer(2)]], device float* mask [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  mask[i] = s[i] == pooled[i] ? 1.0f : 0.0f;
}

// Scores with everything near a current maximum (pooled_mask > 0) set to zero.
kernel void k_nms_suppress(device const float* s [[buffer(0)]], device const float* pooled_mask [[buffer(1)]],
                           constant int& n [[buffer(2)]], device float* supp_scores [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  supp_scores[i] = pooled_mask[i] > 0 ? 0.0f : s[i];
}

// Adds the maxima of the suppressed scores to the mask.
kernel void k_nms_update(device const float* supp_scores [[buffer(0)]], device const float* pooled_supp [[buffer(1)]],
                         device const float* pooled_mask [[buffer(2)]], constant int& n [[buffer(3)]],
                         device float* mask [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  bool suppressed = pooled_mask[i] > 0;
  if (supp_scores[i] == pooled_supp[i] && !suppressed) mask[i] = 1.0f;
}

// nms = score at maxima away from the border, zero elsewhere.
kernel void k_nms_final(device const float* s [[buffer(0)]], device const float* mask [[buffer(1)]],
                        constant int& w [[buffer(2)]], constant int& h [[buffer(3)]], constant int& r [[buffer(4)]],
                        device float* nms [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= w * h) return;
  int y = i / w, x = i % w;
  bool border = x < r || y < r || x >= w - r || y >= h - r;
  nms[i] = (mask[i] > 0 && !border) ? s[i] : 0.0f;
}

inline bool selected(float v, int mode, float th) { return mode == kGreater ? v > th : v == th; }

// Ordered compaction of {i : selected(v[i])}. The exclusive prefix sum over the block counts runs on
// the host between the two kernels (the buffers are shared memory).
// 1. One thread per block of kScanBlock elements counts the selected elements.
kernel void k_select_count(device const float* v [[buffer(0)]], constant int& n [[buffer(1)]],
                           constant int& mode [[buffer(2)]], constant float& th [[buffer(3)]],
                           device uint* counts [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int b = int(gid), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  uint count = 0;
  for (int i = begin; i < end; ++i) count += selected(v[i], mode, th) ? 1u : 0u;
  counts[b] = count;
}
// 2. One thread per block writes its indices at the block offset; only the first `limit` are kept.
kernel void k_select_write(device const float* v [[buffer(0)]], constant int& n [[buffer(1)]],
                           constant int& mode [[buffer(2)]], constant float& th [[buffer(3)]],
                           device const uint* offsets [[buffer(4)]], constant uint& limit [[buffer(5)]],
                           device int* out [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int b = int(gid), begin = b * kScanBlock;
  if (begin >= n) return;
  int end = min(n, begin + kScanBlock);
  uint pos = offsets[b];
  for (int i = begin; i < end && pos < limit; ++i)
    if (selected(v[i], mode, th)) out[pos++] = i;
}

// Radix select: histogram of 16 bits of the float bits, over the values whose higher bits equal `prefix`.
// Values are non-negative, so the bit patterns order like the floats.
kernel void k_histogram(device const float* v [[buffer(0)]], constant int& n [[buffer(1)]],
                        constant uint& mask [[buffer(2)]], constant uint& prefix [[buffer(3)]],
                        constant int& shift [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  uint bits = as_type<uint>(v[i]);
  if ((bits & mask) == prefix) atomic_fetch_add_explicit(&hist[(bits >> shift) & 65535u], 1u, memory_order_relaxed);
}

// Rank sort of k indices by (value descending, index ascending). One thread per element counts the
// elements that come before it; the keys are distinct, so every rank is used exactly once.
kernel void k_rank_sort(device const float* v [[buffer(0)]], device const int* idx [[buffer(1)]],
                        constant int& k [[buffer(2)]], device int* sorted [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
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
kernel void k_refine(device const float* s [[buffer(0)]], constant int& w [[buffer(1)]], constant int& h [[buffer(2)]],
                     device const int* idx [[buffer(3)]], constant int& n [[buffer(4)]], device float* xy_norm [[buffer(5)]],
                     device float* score [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  const float temperature = 0.1f;
  const float wh[2] = {float(w - 1), float(h - 1)};
  const int x = idx[i] % w, y = idx[i] / w;
  float patch[25], max_v = -INFINITY;
  for (int dy = -kRadius; dy <= kRadius; ++dy)
    for (int dx = -kRadius; dx <= kRadius; ++dx) {
      int px = x + dx, py = y + dy;
      float v = (px >= 0 && px < w && py >= 0 && py < h) ? s[py * w + px] : 0.0f;
      patch[(dy + kRadius) * 5 + dx + kRadius] = v;
      max_v = fmax(max_v, v);
    }
  float sum = 0, sx = 0, sy = 0;
  for (int j = 0; j < 25; ++j) {
    float e = precise::exp((patch[j] - max_v) / temperature);
    sum += e;
    sx += e * float(j % 5 - kRadius);
    sy += e * float(j / 5 - kRadius);
  }
  // Normalized to [-1, 1]; the descriptor head truncates the pixel position recovered from this value,
  // so the float32 operation order here and in the descriptor kernels follows the original exactly.
  const float xy[2] = {float(x) + sx / sum, float(y) + sy / sum};
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
// Only the feature map `f` is `real`; the head's weights and buffers are float in both modes.
// ---------------------------------------------------------------------------------------------

// Offsets, first layer: KxK convolution on the patch centred on the truncated keypoint (zeros outside
// the map), then SELU. One thread per (keypoint, output channel); o0[n][2M].
kernel void k_sddh_offset0(device const real* f [[buffer(0)]], constant int& c [[buffer(1)]],
                           constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                           device const float* xy_norm [[buffer(4)]], constant int& n [[buffer(5)]],
                           constant int& K [[buffer(6)]], device const float* weight [[buffer(7)]],
                           device const float* bias [[buffer(8)]], constant int& m2 [[buffer(9)]],
                           device float* o0 [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2, hw = h * w, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * float(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * float(h - 1);
  const int ix = int(kx), iy = int(ky);  // truncation
  device const float* wk = weight + size_t(j) * c * K * K;
  float acc = 0;
  for (int ch = 0; ch < c; ++ch) {
    device const real* plane = f + size_t(ch) * hw;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) {
        int py = iy - lo + a, px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) acc += wk[(ch * K + a) * K + b] * float(plane[py * w + px]);
      }
  }
  o0[t] = selu(acc + bias[j]);
}

// Offsets, second layer: 1x1 convolution, clamped. o1[n][2M] (x offsets first, then y).
kernel void k_sddh_offset1(device const float* o0 [[buffer(0)]], constant int& n [[buffer(1)]],
                           constant int& m2 [[buffer(2)]], device const float* weight [[buffer(3)]],
                           device const float* bias [[buffer(4)]], constant float& max_offset [[buffer(5)]],
                           device float* o1 [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2;
  device const float* wk = weight + size_t(j) * m2;
  device const float* in = o0 + size_t(i) * m2;
  float acc = 0;
  for (int q = 0; q < m2; ++q) acc += wk[q] * in[q];
  o1[t] = clampf(acc + bias[j], -max_offset, max_offset);
}

// Bilinear samples of the feature map at keypoint + offset. One thread per (keypoint, position);
// sampled[n][M][C].
kernel void k_sddh_sample(device const real* f [[buffer(0)]], constant int& c [[buffer(1)]],
                          constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                          device const float* xy_norm [[buffer(4)]], device const float* o1 [[buffer(5)]],
                          constant int& n [[buffer(6)]], constant int& M [[buffer(7)]], device float* sampled [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * M) return;
  int i = t / M, m = t % M, hw = h * w;
  const float wh[2] = {float(w - 1), float(h - 1)};
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * wh[0];
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * wh[1];
  device const float* off = o1 + size_t(i) * 2 * M;
  float gx = 2.0f * (kx + off[m]) / wh[0] - 1.0f, gy = 2.0f * (ky + off[M + m]) / wh[1] - 1.0f;
  gx = ((gx + 1.0f) / 2.0f) * wh[0], gy = ((gy + 1.0f) / 2.0f) * wh[1];
  int bi[4];
  float bw[4];
  bilinear_setup(gx, gy, w, h, bi, bw);
  device float* out = sampled + size_t(t) * c;
  for (int ch = 0; ch < c; ++ch) out[ch] = bilinear_apply(f + size_t(ch) * hw, bi, bw);
}

// 1x1 convolution + SELU on the samples. One thread per (keypoint, position, output channel).
kernel void k_sddh_sf(device const float* sampled [[buffer(0)]], constant int& total [[buffer(1)]],
                      constant int& c [[buffer(2)]], device const float* weight [[buffer(3)]],
                      device float* feat [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= total) return;
  int oc = t % c;
  device const float* wk = weight + size_t(oc) * c;
  device const float* in = sampled + size_t(t - oc);
  float acc = 0;
  for (int q = 0; q < c; ++q) acc += wk[q] * in[q];
  feat[t] = selu(acc);
}

// Aggregation with one CxD matrix per position. One thread per (keypoint, descriptor element).
kernel void k_sddh_aggregate(device const float* feat [[buffer(0)]], constant int& n [[buffer(1)]],
                             constant int& mc [[buffer(2)]], constant int& dim [[buffer(3)]],
                             device const float* weight [[buffer(4)]], device float* desc [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * dim) return;
  int i = t / dim, j = t % dim;
  device const float* in = feat + size_t(i) * mc;
  float d = 0;
  for (int q = 0; q < mc; ++q) d += in[q] * weight[size_t(q) * dim + j];
  desc[t] = d;
}

// k_sddh_sf, T output channels per thread; weights tap-major (input channel major), same order per output.
// One thread per (keypoint, position, block of T output channels).
template <int T>
kernel void k_sddh_sf_tile(device const float* sampled [[buffer(0)]], constant int& total [[buffer(1)]],
                           constant int& c [[buffer(2)]], constant packed_float4* weight [[buffer(3)]],
                           device float* feat [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const int blocks = (c + T - 1) / T, i = int(gid);
  if (i >= total / c * blocks) return;
  const int s = i / blocks, ob = i % blocks, oc0 = ob * T;
  device const float* in = sampled + size_t(s) * c;
  float4 acc[T / 4];
  for (int q = 0; q < T / 4; ++q) acc[q] = float4(0.0f);
  for (int ch = 0; ch < c; ++ch) {
    const float4 v = float4(in[ch]);
    constant packed_float4* wp = weight + (ch * blocks + ob) * (T / 4);
    for (int q = 0; q < T / 4; ++q) JKP_MAC(acc[q], float4(wp[q]), v);
  }
  for (int t = 0; t < T; ++t)
    if (oc0 + t < c) feat[size_t(s) * c + oc0 + t] = selu(acc[t / 4][t % 4]);
}
template [[host_name("k_sddh_sf_t16")]] kernel void k_sddh_sf_tile<16>(device const float*, constant int&, constant int&,
                                                                    constant packed_float4*, device float*, uint);

// k_sddh_aggregate, T descriptor elements per thread (dim is a multiple of T); weight[M*C][dim] as it is.
// One thread per (keypoint, block of T elements).
template <int T>
kernel void k_sddh_aggregate_tile(device const float* feat [[buffer(0)]], constant int& n [[buffer(1)]],
                                  constant int& mc [[buffer(2)]], constant int& dim [[buffer(3)]],
                                  constant packed_float4* weight [[buffer(4)]], device float* desc [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
  const int blocks = dim / T, t = int(gid);
  if (t >= n * blocks) return;
  const int i = t / blocks, j0 = (t % blocks) * T;
  device const float* in = feat + size_t(i) * mc;
  float4 acc[T / 4];
  for (int q = 0; q < T / 4; ++q) acc[q] = float4(0.0f);
  for (int k = 0; k < mc; ++k) {
    const float4 v = float4(in[k]);
    constant packed_float4* wp = weight + (k * dim + j0) / 4;
    for (int q = 0; q < T / 4; ++q) JKP_MAC(acc[q], v, float4(wp[q]));
  }
  for (int q = 0; q < T; ++q) desc[size_t(i) * dim + j0 + q] = acc[q / 4][q % 4];
}
template [[host_name("k_sddh_aggregate_t16")]] kernel void k_sddh_aggregate_tile<16>(
    device const float*, constant int&, constant int&, constant int&, constant packed_float4*, device float*, uint);

// L2 normalization of each descriptor. One thread per keypoint.
kernel void k_normalize_rows(device float* desc [[buffer(0)]], constant int& n [[buffer(1)]],
                             constant int& dim [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  device float* d = desc + size_t(i) * dim;
  float norm = 0;
  for (int j = 0; j < dim; ++j) norm += d[j] * d[j];
  norm = fmax(precise::sqrt(norm), 1e-12f);
  for (int j = 0; j < dim; ++j) d[j] /= norm;
}

// ---------------------------------------------------------------------------------------------
// Descriptor head on a pyramid: the feature map is the L2-normalized pyramid, cropped to w x h at
// (a.left, a.top). These kernels normalize only the pixels the keypoints touch; they replace
// k_sddh_offset0 and k_sddh_sample and give the same values as those do on the stored feature map.
// ---------------------------------------------------------------------------------------------

// Normalized KxK patch around the truncated keypoint. One thread per (keypoint, patch pixel);
// patch[n][K*K][C], left untouched (and unused) for pixels outside the map.
kernel void k_sddh_patch_pyramid(device const real* p0 [[buffer(0)]], device const real* p1 [[buffer(1)]],
                                 device const real* p2 [[buffer(2)]], device const real* p3 [[buffer(3)]],
                                 constant PyramidArgs& a [[buffer(4)]], constant int& h [[buffer(5)]],
                                 constant int& w [[buffer(6)]], device const float* xy_norm [[buffer(7)]],
                                 constant int& n [[buffer(8)]], constant int& K [[buffer(9)]],
                                 device float* patch [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  const int t = int(gid), kk = K * K;
  if (t >= n * kk) return;
  const int i = t / kk, q = t % kk, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * float(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * float(h - 1);
  const int px = int(kx) - lo + q % K, py = int(ky) - lo + q / K;  // truncation
  if (px < 0 || px >= w || py < 0 || py >= h) return;
  const int X = px + a.left, Y = py + a.top;
  Taps taps[4];
  for (int p = 0; p < 4; ++p) taps[p] = part_taps(a, p, X, Y);
  const float norm = pyramid_norm(p0, p1, p2, p3, a, taps);
  device float* out = patch + size_t(t) * (a.c[0] + a.c[1] + a.c[2] + a.c[3]);
  for (int ch = 0; ch < a.c[0]; ++ch) *out++ = part_value(p0, a, 0, ch, taps[0]) / norm;
  for (int ch = 0; ch < a.c[1]; ++ch) *out++ = part_value(p1, a, 1, ch, taps[1]) / norm;
  for (int ch = 0; ch < a.c[2]; ++ch) *out++ = part_value(p2, a, 2, ch, taps[2]) / norm;
  for (int ch = 0; ch < a.c[3]; ++ch) *out++ = part_value(p3, a, 3, ch, taps[3]) / norm;
}

// k_sddh_offset0 on the patches of k_sddh_patch_pyramid.
kernel void k_sddh_offset0_patch(device const float* patch [[buffer(0)]], constant int& c [[buffer(1)]],
                                 constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                                 device const float* xy_norm [[buffer(4)]], constant int& n [[buffer(5)]],
                                 constant int& K [[buffer(6)]], device const float* weight [[buffer(7)]],
                                 device const float* bias [[buffer(8)]], constant int& m2 [[buffer(9)]],
                                 device float* o0 [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * m2) return;
  int i = t / m2, j = t % m2, lo = (K - 1) / 2;
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * float(w - 1);
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * float(h - 1);
  const int ix = int(kx), iy = int(ky);  // truncation
  device const float* wk = weight + size_t(j) * c * K * K;
  device const float* in = patch + size_t(i) * K * K * c;
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
constant int kMaxPyramidC = 128;

inline void pyramid_read_part(device const real* m, constant PyramidArgs& a, int p, Taps taps, thread float* vals,
                              thread int& pos, thread float& norm) {
  for (int ch = 0; ch < a.c[p]; ++ch) {
    const float v = part_value(m, a, p, ch, taps);
    vals[pos++] = v;
    norm += v * v;
  }
}

kernel void k_sddh_sample_pyramid(device const real* p0 [[buffer(0)]], device const real* p1 [[buffer(1)]],
                                  device const real* p2 [[buffer(2)]], device const real* p3 [[buffer(3)]],
                                  constant PyramidArgs& a [[buffer(4)]], constant int& h [[buffer(5)]],
                                  constant int& w [[buffer(6)]], device const float* xy_norm [[buffer(7)]],
                                  device const float* o1 [[buffer(8)]], constant int& n [[buffer(9)]],
                                  constant int& M [[buffer(10)]], device float* sampled [[buffer(11)]],
                                  uint gid [[thread_position_in_grid]]) {
  int t = int(gid);
  if (t >= n * M) return;
  int i = t / M, m = t % M;
  const float wh[2] = {float(w - 1), float(h - 1)};
  const float kx = (xy_norm[2 * i] / 2.0f + 0.5f) * wh[0];
  const float ky = (xy_norm[2 * i + 1] / 2.0f + 0.5f) * wh[1];
  device const float* off = o1 + size_t(i) * 2 * M;
  float gx = 2.0f * (kx + off[m]) / wh[0] - 1.0f, gy = 2.0f * (ky + off[M + m]) / wh[1] - 1.0f;
  gx = ((gx + 1.0f) / 2.0f) * wh[0], gy = ((gy + 1.0f) / 2.0f) * wh[1];
  int bi[4];
  float bw[4];
  bilinear_setup(gx, gy, w, h, bi, bw);
  const int c = a.c[0] + a.c[1] + a.c[2] + a.c[3];
  float vals[kMaxPyramidC], acc[kMaxPyramidC];
  // A corner outside the map has weight 0 and index 0, as in bilinear_apply: it adds 0 * (pixel 0).
  for (int k = 0; k < 4; ++k) {
    const int X = bi[k] % w + a.left, Y = bi[k] / w + a.top;
    int pos = 0;
    float norm = 0;
    pyramid_read_part(p0, a, 0, part_taps(a, 0, X, Y), vals, pos, norm);
    pyramid_read_part(p1, a, 1, part_taps(a, 1, X, Y), vals, pos, norm);
    pyramid_read_part(p2, a, 2, part_taps(a, 2, X, Y), vals, pos, norm);
    pyramid_read_part(p3, a, 3, part_taps(a, 3, X, Y), vals, pos, norm);
    norm = fmax(precise::sqrt(norm), 1e-12f);
    for (int ch = 0; ch < c; ++ch) {
      const float v = bw[k] * (vals[ch] / norm);
      acc[ch] = k == 0 ? v : acc[ch] + v;
    }
  }
  device float* out = sampled + size_t(t) * c;
  for (int ch = 0; ch < c; ++ch) out[ch] = acc[ch];
}
