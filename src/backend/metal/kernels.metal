// Metal backend kernels: one compute kernel per operation, FP32, CHW layout, same graph as the CPU backend.
//
// Each kernel is a port of the CUDA kernel of the same name (src/backend/cuda/cuda_ops.cu) and keeps the
// float32 operation order of src/backend/cpu/cpu_ops.cpp. Contraction into fused multiply-add is off and
// the library is compiled in safe math mode with precise functions (see metal_ops.mm), so the backends
// differ only through exp() (SELU, sigmoid, soft-argmax).
//
// The source is compiled at run time (newLibraryWithSource), so building needs no Metal toolchain.
// Scalar arguments are passed one per buffer index with setBytes, in the order of the CUDA signatures.
#include <metal_stdlib>
using namespace metal;

#pragma METAL fp contract(off)

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
inline float bilinear_apply(device const float* plane, thread const int* idx, thread const float* wt) {
  return wt[0] * plane[idx[0]] + wt[1] * plane[idx[1]] + wt[2] * plane[idx[2]] + wt[3] * plane[idx[3]];
}

// ---------------------------------------------------------------------------------------------
// Dense kernels. One thread per output element unless noted.
// ---------------------------------------------------------------------------------------------

kernel void k_image(device const uchar* src [[buffer(0)]], constant int& stride [[buffer(1)]],
                    constant int& channels [[buffer(2)]], constant int& w [[buffer(3)]], constant int& h [[buffer(4)]],
                    device float* dst [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), hw = h * w;
  if (i >= 3 * hw) return;
  int c = i / hw, y = (i % hw) / w, x = i % w;
  int src_c = channels == 1 ? 0 : c;
  dst[i] = float(src[size_t(y) * stride + x * channels + src_c]) / 255.0f;
}

kernel void k_pad_replicate(device const float* src [[buffer(0)]], constant int& c [[buffer(1)]],
                            constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                            constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                            constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]],
                            device float* dst [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  int sy = min(max(y - top, 0), h - 1), sx = min(max(x - left, 0), w - 1);
  dst[i] = src[(size_t(ch) * h + sy) * w + sx];
}

kernel void k_crop(device const float* src [[buffer(0)]], constant int& c [[buffer(1)]], constant int& h [[buffer(2)]],
                   constant int& w [[buffer(3)]], constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                   constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]], device float* dst [[buffer(8)]],
                   uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  dst[i] = src[(size_t(ch) * h + y + top) * w + x + left];
}

kernel void k_conv2d(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]],
                     device const float* bias [[buffer(2)]], constant int& has_bias [[buffer(3)]],
                     constant int& cin [[buffer(4)]], constant int& cout [[buffer(5)]], constant int& h [[buffer(6)]],
                     constant int& w [[buffer(7)]], constant int& k [[buffer(8)]], constant int& act [[buffer(9)]],
                     device float* out [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), hw = h * w;
  if (i >= cout * hw) return;
  int oc = i / hw, y = (i % hw) / w, xo = i % w, pad = k / 2;
  float acc = has_bias ? bias[oc] : 0.0f;
  device const float* wk = weight + size_t(oc) * cin * k * k;
  for (int ic = 0; ic < cin; ++ic, wk += k * k) {
    device const float* in = x + size_t(ic) * hw;
    for (int ky = 0; ky < k; ++ky) {
      int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < k; ++kx) {
        int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        acc += wk[ky * k + kx] * in[sy * w + sx];
      }
    }
  }
  out[i] = activate(acc, act);
}

// Convolution, T output channels per thread for one pixel, KxK kernel known at compile time. Each input
// value is read once for all T outputs instead of once per output, and each output accumulates in the
// same order as k_conv2d, so the results are identical. Weights are uniform across the threads of a
// SIMD group and read through the constant address space. Grid: ceil(cout / T) * h * w, pixel fastest.
template <int T, int K>
kernel void k_conv2d_tile(device const float* x [[buffer(0)]], constant float* weight [[buffer(1)]],
                          constant float* bias [[buffer(2)]], constant int& has_bias [[buffer(3)]],
                          constant int& cin [[buffer(4)]], constant int& cout [[buffer(5)]], constant int& h [[buffer(6)]],
                          constant int& w [[buffer(7)]], constant int& act [[buffer(8)]], device float* out [[buffer(9)]],
                          uint gid [[thread_position_in_grid]]) {
  const int hw = h * w, blocks = (cout + T - 1) / T;
  const int i = int(gid);
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, y = pix / w, xo = pix % w, pad = K / 2, oc0 = ob * T;
  const int wstride = cin * K * K;  // weights per output channel
  // Channels past cout (last block of an odd cout) repeat the last channel and are not stored.
  int woff[T];
  float acc[T];
  for (int t = 0; t < T; ++t) {
    const int oc = min(oc0 + t, cout - 1);
    woff[t] = oc * wstride;
    acc[t] = has_bias ? bias[oc] : 0.0f;
  }
  for (int ic = 0; ic < cin; ++ic) {
    device const float* in = x + size_t(ic) * hw;
    const int wic = ic * K * K;
    for (int ky = 0; ky < K; ++ky) {
      const int sy = y + ky - pad;
      if (sy < 0 || sy >= h) continue;
      for (int kx = 0; kx < K; ++kx) {
        const int sx = xo + kx - pad;
        if (sx < 0 || sx >= w) continue;
        const float v = in[sy * w + sx];
        const int tap = wic + ky * K + kx;
        for (int t = 0; t < T; ++t) acc[t] += weight[woff[t] + tap] * v;
      }
    }
  }
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) out[size_t(oc0 + t) * hw + pix] = activate(acc[t], act);
}

#define JKP_CONV_TILE(T, K)                                                                                \
  template [[host_name("k_conv2d_t" #T "_k" #K)]] kernel void k_conv2d_tile<T, K>(                          \
      device const float*, constant float*, constant float*, constant int&, constant int&, constant int&, \
      constant int&, constant int&, constant int&, device float*, uint);
JKP_CONV_TILE(4, 1)
JKP_CONV_TILE(8, 1)
JKP_CONV_TILE(16, 1)
JKP_CONV_TILE(32, 1)
JKP_CONV_TILE(4, 3)
JKP_CONV_TILE(8, 3)
JKP_CONV_TILE(16, 3)
JKP_CONV_TILE(32, 3)

// Deformable convolution, step 1. One thread per (kernel tap, pixel): samples every input channel at
// the offset position. col[(ic * 9 + tap) * hw + pixel].
kernel void k_deform_sample(device const float* x [[buffer(0)]], device const float* off [[buffer(1)]],
                            constant int& cin [[buffer(2)]], constant int& h [[buffer(3)]], constant int& w [[buffer(4)]],
                            constant float& max_offset [[buffer(5)]], device float* col [[buffer(6)]],
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
  for (int ic = 0; ic < cin; ++ic) col[(size_t(ic) * 9 + k) * hw + pix] = bilinear_apply(x + size_t(ic) * hw, idx, wt);
}

// Deformable convolution, step 2: weighted sum over the sampled columns.
kernel void k_deform_sum(device const float* col [[buffer(0)]], device const float* weight [[buffer(1)]],
                         device const float* bias [[buffer(2)]], constant int& taps [[buffer(3)]],
                         constant int& cout [[buffer(4)]], constant int& hw [[buffer(5)]], constant int& act [[buffer(6)]],
                         device float* out [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= cout * hw) return;
  int oc = i / hw, pix = i % hw;
  device const float* wk = weight + size_t(oc) * taps;
  float acc = bias[oc];
  for (int j = 0; j < taps; ++j) acc += wk[j] * col[size_t(j) * hw + pix];
  out[i] = activate(acc, act);
}

// Deformable convolution, step 2, T output channels per thread (same order per output as k_deform_sum).
template <int T>
kernel void k_deform_sum_tile(device const float* col [[buffer(0)]], constant float* weight [[buffer(1)]],
                              constant float* bias [[buffer(2)]], constant int& taps [[buffer(3)]],
                              constant int& cout [[buffer(4)]], constant int& hw [[buffer(5)]],
                              constant int& act [[buffer(6)]], device float* out [[buffer(7)]],
                              uint gid [[thread_position_in_grid]]) {
  const int blocks = (cout + T - 1) / T;
  const int i = int(gid);
  if (i >= blocks * hw) return;
  const int ob = i / hw, pix = i % hw, oc0 = ob * T;
  int woff[T];
  float acc[T];
  for (int t = 0; t < T; ++t) {
    const int oc = min(oc0 + t, cout - 1);
    woff[t] = oc * taps;
    acc[t] = bias[oc];
  }
  for (int j = 0; j < taps; ++j) {
    const float v = col[size_t(j) * hw + pix];
    for (int t = 0; t < T; ++t) acc[t] += weight[woff[t] + j] * v;
  }
  for (int t = 0; t < T; ++t)
    if (oc0 + t < cout) out[size_t(oc0 + t) * hw + pix] = activate(acc[t], act);
}

#define JKP_DEFORM_TILE(T)                                                                                   \
  template [[host_name("k_deform_sum_t" #T)]] kernel void k_deform_sum_tile<T>(                              \
      device const float*, constant float*, constant float*, constant int&, constant int&, constant int&,   \
      constant int&, device float*, uint);
JKP_DEFORM_TILE(4)
JKP_DEFORM_TILE(8)
JKP_DEFORM_TILE(16)
JKP_DEFORM_TILE(32)

kernel void k_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], constant int& n [[buffer(2)]],
                  constant int& act [[buffer(3)]], device float* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  out[i] = activate(a[i] + b[i], act);
}

kernel void k_avg_pool(device const float* x [[buffer(0)]], constant int& c [[buffer(1)]], constant int& w [[buffer(2)]],
                       constant int& oh [[buffer(3)]], constant int& ow [[buffer(4)]], constant int& k [[buffer(5)]],
                       device float* out [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  device const float* in = x + size_t(ch) * (oh * k) * w;
  float sum = 0;
  for (int j = 0; j < k; ++j)
    for (int q = 0; q < k; ++q) sum += in[(y * k + j) * w + xo * k + q];
  out[i] = sum / float(k * k);
}

// Bilinear, align_corners = true: source position = dst * (in - 1) / (out - 1); sx, sy are those ratios.
kernel void k_upsample(device const float* x [[buffer(0)]], constant int& c [[buffer(1)]], constant int& h [[buffer(2)]],
                       constant int& w [[buffer(3)]], constant int& oh [[buffer(4)]], constant int& ow [[buffer(5)]],
                       constant float& sx [[buffer(6)]], constant float& sy [[buffer(7)]], device float* out [[buffer(8)]],
                       uint gid [[thread_position_in_grid]]) {
  int i = int(gid), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  float src_x = sx * float(xo), src_y = sy * float(y);
  int x0 = min(int(src_x), w - 1), y0 = min(int(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  float w1 = src_x - float(x0), w0 = 1.0f - w1;
  float h1 = src_y - float(y0), h0 = 1.0f - h1;
  device const float* r0 = x + (size_t(ch) * h + y0) * w;
  device const float* r1 = x + (size_t(ch) * h + y1) * w;
  out[i] = h0 * (w0 * r0[x0] + w1 * r0[x1]) + h1 * (w0 * r1[x0] + w1 * r1[x1]);
}

kernel void k_copy(device const float* src [[buffer(0)]], constant int& n [[buffer(1)]], device float* dst [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  dst[i] = src[i];
}

// One thread per pixel.
kernel void k_l2_normalize(device const float* x [[buffer(0)]], constant int& c [[buffer(1)]],
                           constant int& hw [[buffer(2)]], device float* out [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= hw) return;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = x[size_t(ch) * hw + i];
    norm += v * v;
  }
  norm = fmax(precise::sqrt(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch) out[size_t(ch) * hw + i] = x[size_t(ch) * hw + i] / norm;
}

// L2 normalization across channels of the cropped region of x (c x h x w) at (left, top); out is
// c x oh x ow. Same arithmetic as k_l2_normalize followed by k_crop. One thread per output pixel.
kernel void k_l2_normalize_crop(device const float* x [[buffer(0)]], constant int& c [[buffer(1)]],
                                constant int& h [[buffer(2)]], constant int& w [[buffer(3)]],
                                constant int& left [[buffer(4)]], constant int& top [[buffer(5)]],
                                constant int& oh [[buffer(6)]], constant int& ow [[buffer(7)]],
                                device float* out [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  const int i = int(gid), ohw = oh * ow;
  if (i >= ohw) return;
  const int hw = h * w, src = (i / ow + top) * w + i % ow + left;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = x[size_t(ch) * hw + src];
    norm += v * v;
  }
  norm = fmax(precise::sqrt(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch) out[size_t(ch) * ohw + i] = x[size_t(ch) * hw + src] / norm;
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

// Radix select: histogram of one byte of the float bits, over the values whose higher bytes equal `prefix`.
// Values are non-negative, so the bit patterns order like the floats.
kernel void k_histogram(device const float* v [[buffer(0)]], constant int& n [[buffer(1)]],
                        constant uint& mask [[buffer(2)]], constant uint& prefix [[buffer(3)]],
                        constant int& shift [[buffer(4)]], device atomic_uint* hist [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i >= n) return;
  uint bits = as_type<uint>(v[i]);
  if ((bits & mask) == prefix) atomic_fetch_add_explicit(&hist[(bits >> shift) & 255u], 1u, memory_order_relaxed);
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
// ---------------------------------------------------------------------------------------------

// Offsets, first layer: KxK convolution on the patch centred on the truncated keypoint (zeros outside
// the map), then SELU. One thread per (keypoint, output channel); o0[n][2M].
kernel void k_sddh_offset0(device const float* f [[buffer(0)]], constant int& c [[buffer(1)]],
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
    device const float* plane = f + size_t(ch) * hw;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) {
        int py = iy - lo + a, px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) acc += wk[(ch * K + a) * K + b] * plane[py * w + px];
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
kernel void k_sddh_sample(device const float* f [[buffer(0)]], constant int& c [[buffer(1)]],
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
