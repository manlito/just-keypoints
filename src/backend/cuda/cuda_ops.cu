// CUDA backend: one hand-written kernel per operation, FP32, CHW layout, same graph as the CPU backend.
//
// Correctness first (PLAN.md phase 2). Every kernel keeps the float32 operation order of
// src/backend/cpu/cpu_ops.cpp and the file is compiled with --fmad=false, so the only source of
// difference from the CPU backend is the exp() implementation (SELU, sigmoid, soft-argmax).
//
// Keypoint selection (threshold, top-k, compaction, ordering) runs on the device; only the selected
// keypoints and their descriptors are copied back to the host.
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

#include "core/ops.h"

namespace jkp {
namespace {

// ---------------------------------------------------------------------------------------------
// Device helpers shared by the kernels
// ---------------------------------------------------------------------------------------------

enum : int { kActNone = 0, kActSelu = 1, kActSigmoid = 2 };
enum : int { kGreater = 0, kEqual = 1 };
constexpr int kRadius = 2;       // NMS and soft-argmax window radius
constexpr int kScanBlock = 256;  // elements handled by one thread in the compaction kernels

__device__ __forceinline__ float selu(float x) {
  const float alpha = 1.6732632423543772848170429916717f, scale = 1.0507009873554804934193349852946f;
  return x > 0 ? scale * x : scale * (alpha * (expf(x) - 1.0f));
}

__device__ __forceinline__ float activate(float v, int act) {
  if (act == kActSelu) return selu(v);
  if (act == kActSigmoid) return 1.0f / (1.0f + expf(-v));
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
__device__ __forceinline__ float bilinear_apply(const float* plane, const int idx[4], const float wt[4]) {
  return wt[0] * plane[idx[0]] + wt[1] * plane[idx[1]] + wt[2] * plane[idx[2]] + wt[3] * plane[idx[3]];
}

__device__ __forceinline__ int thread_index() { return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); }

// ---------------------------------------------------------------------------------------------
// Dense kernels. One thread per output element unless noted.
// ---------------------------------------------------------------------------------------------

__global__ void k_image(const uint8_t* src, int stride, int channels, int w, int h, float* dst) {
  int i = thread_index(), hw = h * w;
  if (i >= 3 * hw) return;
  int c = i / hw, y = (i % hw) / w, x = i % w;
  int src_c = channels == 1 ? 0 : c;
  dst[i] = static_cast<float>(src[static_cast<size_t>(y) * stride + x * channels + src_c]) / 255.0f;
}

__global__ void k_pad_replicate(const float* src, int c, int h, int w, int left, int top, int oh, int ow, float* dst) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  int sy = min(max(y - top, 0), h - 1), sx = min(max(x - left, 0), w - 1);
  dst[i] = src[(ch * h + sy) * w + sx];
}

__global__ void k_crop(const float* src, int c, int h, int w, int left, int top, int oh, int ow, float* dst) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, x = i % ow;
  dst[i] = src[(ch * h + y + top) * w + x + left];
}

__global__ void k_conv2d(const float* x, const float* weight, const float* bias, int cin, int cout, int h, int w,
                         int k, int act, float* out) {
  int i = thread_index(), hw = h * w;
  if (i >= cout * hw) return;
  int oc = i / hw, y = (i % hw) / w, xo = i % w, pad = k / 2;
  float acc = bias ? bias[oc] : 0.0f;
  const float* wk = weight + static_cast<size_t>(oc) * cin * k * k;
  for (int ic = 0; ic < cin; ++ic, wk += k * k) {
    const float* in = x + static_cast<size_t>(ic) * hw;
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

// Deformable convolution, step 1. One thread per (kernel tap, pixel): samples every input channel at
// the offset position. col[(ic * 9 + tap) * hw + pixel].
__global__ void k_deform_sample(const float* x, const float* off, int cin, int h, int w, float max_offset,
                                float* col) {
  int i = thread_index(), hw = h * w;
  if (i >= 9 * hw) return;
  int k = i / hw, pix = i % hw, y = pix / w, xo = pix % w;
  float off_y = off[static_cast<size_t>(2 * k) * hw + pix], off_x = off[static_cast<size_t>(2 * k + 1) * hw + pix];
  float py = static_cast<float>(y - 1 + k / 3) + clampf(off_y, -max_offset, max_offset);
  float px = static_cast<float>(xo - 1 + k % 3) + clampf(off_x, -max_offset, max_offset);
  int idx[4] = {0, 0, 0, 0};
  float wt[4] = {0, 0, 0, 0};
  if (py > -1 && py < h && px > -1 && px < w) bilinear_setup(px, py, w, h, idx, wt);
  for (int ic = 0; ic < cin; ++ic)
    col[(static_cast<size_t>(ic) * 9 + k) * hw + pix] = bilinear_apply(x + static_cast<size_t>(ic) * hw, idx, wt);
}

// Deformable convolution, step 2: weighted sum over the sampled columns.
__global__ void k_deform_sum(const float* col, const float* weight, const float* bias, int taps, int cout, int hw,
                             int act, float* out) {
  int i = thread_index();
  if (i >= cout * hw) return;
  int oc = i / hw, pix = i % hw;
  const float* wk = weight + static_cast<size_t>(oc) * taps;
  float acc = bias[oc];
  for (int j = 0; j < taps; ++j) acc += wk[j] * col[static_cast<size_t>(j) * hw + pix];
  out[i] = activate(acc, act);
}

__global__ void k_add(const float* a, const float* b, int n, int act, float* out) {
  int i = thread_index();
  if (i >= n) return;
  out[i] = activate(a[i] + b[i], act);
}

__global__ void k_avg_pool(const float* x, int c, int w, int oh, int ow, int k, float* out) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  const float* in = x + static_cast<size_t>(ch) * (oh * k) * w;
  float sum = 0;
  for (int j = 0; j < k; ++j)
    for (int q = 0; q < k; ++q) sum += in[(y * k + j) * w + xo * k + q];
  out[i] = sum / static_cast<float>(k * k);
}

// Bilinear, align_corners = true: source position = dst * (in - 1) / (out - 1); sx, sy are those ratios.
__global__ void k_upsample(const float* x, int c, int h, int w, int oh, int ow, float sx, float sy, float* out) {
  int i = thread_index(), ohw = oh * ow;
  if (i >= c * ohw) return;
  int ch = i / ohw, y = (i % ohw) / ow, xo = i % ow;
  float src_x = sx * static_cast<float>(xo), src_y = sy * static_cast<float>(y);
  int x0 = min(static_cast<int>(src_x), w - 1), y0 = min(static_cast<int>(src_y), h - 1);
  int x1 = x0 + (x0 < w - 1 ? 1 : 0), y1 = y0 + (y0 < h - 1 ? 1 : 0);
  float w1 = src_x - static_cast<float>(x0), w0 = 1.0f - w1;
  float h1 = src_y - static_cast<float>(y0), h0 = 1.0f - h1;
  const float* r0 = x + (static_cast<size_t>(ch) * h + y0) * w;
  const float* r1 = x + (static_cast<size_t>(ch) * h + y1) * w;
  out[i] = h0 * (w0 * r0[x0] + w1 * r0[x1]) + h1 * (w0 * r1[x0] + w1 * r1[x1]);
}

// One thread per pixel.
__global__ void k_l2_normalize(const float* x, int c, int hw, float* out) {
  int i = thread_index();
  if (i >= hw) return;
  float norm = 0;
  for (int ch = 0; ch < c; ++ch) {
    float v = x[static_cast<size_t>(ch) * hw + i];
    norm += v * v;
  }
  norm = fmaxf(sqrtf(norm), 1e-12f);
  for (int ch = 0; ch < c; ++ch) out[static_cast<size_t>(ch) * hw + i] = x[static_cast<size_t>(ch) * hw + i] / norm;
}

// ---------------------------------------------------------------------------------------------
// Detection kernels
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

// Radix select: histogram of one byte of the float bits, over the values whose higher bytes equal `prefix`.
// Values are non-negative, so the bit patterns order like the floats.
__global__ void k_histogram(const float* v, int n, unsigned mask, unsigned prefix, int shift, unsigned* hist) {
  int i = thread_index();
  if (i >= n) return;
  unsigned bits = __float_as_uint(v[i]);
  if ((bits & mask) == prefix) atomicAdd(&hist[(bits >> shift) & 255u], 1u);
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
// ---------------------------------------------------------------------------------------------

// Offsets, first layer: KxK convolution on the patch centred on the truncated keypoint (zeros outside
// the map), then SELU. One thread per (keypoint, output channel); o0[n][2M].
__global__ void k_sddh_offset0(const float* f, int c, int h, int w, const float* xy_norm, int n, int K,
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
    const float* plane = f + static_cast<size_t>(ch) * hw;
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) {
        int py = iy - lo + a, px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) acc += wk[(ch * K + a) * K + b] * plane[py * w + px];
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
__global__ void k_sddh_sample(const float* f, int c, int h, int w, const float* xy_norm, const float* o1, int n,
                              int M, float* sampled) {
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

// 1x1 convolution + SELU on the samples. One thread per (keypoint, position, output channel).
__global__ void k_sddh_sf(const float* sampled, int total, int c, const float* weight, float* feat) {
  int t = thread_index();
  if (t >= total) return;
  int oc = t % c;
  const float* wk = weight + static_cast<size_t>(oc) * c;
  const float* in = sampled + static_cast<size_t>(t - oc);
  float acc = 0;
  for (int q = 0; q < c; ++q) acc += wk[q] * in[q];
  feat[t] = selu(acc);
}

// Aggregation with one CxD matrix per position. One thread per (keypoint, descriptor element).
__global__ void k_sddh_aggregate(const float* feat, int n, int mc, int dim, const float* weight, float* desc) {
  int t = thread_index();
  if (t >= n * dim) return;
  int i = t / dim, j = t % dim;
  const float* in = feat + static_cast<size_t>(i) * mc;
  float d = 0;
  for (int q = 0; q < mc; ++q) d += in[q] * weight[static_cast<size_t>(q) * dim + j];
  desc[t] = d;
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

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(status));
}
#define JKP_CUDA_CHECK(call) check((call), #call)

constexpr int kBlock = 256;

// Launches `kernel` with at least n threads.
#define JKP_LAUNCH(kernel, n, ...)                                                   \
  do {                                                                               \
    if ((n) > 0) {                                                                   \
      kernel<<<(static_cast<unsigned>(n) + kBlock - 1) / kBlock, kBlock>>>(__VA_ARGS__); \
      check(cudaGetLastError(), #kernel);                                            \
    }                                                                                \
  } while (0)

// Element count as int; the kernels index with 32-bit integers.
int count(long long n) {
  if (n < 0 || n > INT_MAX) throw std::runtime_error("cuda: tensor too large");
  return static_cast<int>(n);
}

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

struct CudaTensor : Tensor {
  Buffer<float> buf;
  CudaTensor(const PoolPtr& pool, int c_, int h_, int w_)
      : buf(pool, static_cast<size_t>(count(static_cast<long long>(c_) * h_ * w_))), device(pool->device()) {
    c = c_, h = h_, w = w_;
  }
  void read(float* dst) const override {
    JKP_CUDA_CHECK(cudaSetDevice(device));
    buf.download(dst, buf.size());
  }
  float* d() const { return buf.get(); }
  int size() const { return static_cast<int>(buf.size()); }
  int device;
};
using CudaPtr = std::shared_ptr<CudaTensor>;

const CudaTensor& dev(const TensorPtr& t) { return static_cast<const CudaTensor&>(*t); }

int act_id(Act act) { return act == Act::Selu ? kActSelu : act == Act::Sigmoid ? kActSigmoid : kActNone; }

class CudaOps : public Ops {
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
      for (auto& kv : params_) cudaFree(kv.second);
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
    JKP_LAUNCH(k_image, out->size(), src.get(), stride, channels, width, height, out->d());
    return out;
  }

  TensorPtr pad_replicate(const TensorPtr& xp, int left, int right, int top, int bottom) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h + top + bottom, x.w + left + right);
    JKP_LAUNCH(k_pad_replicate, out->size(), x.d(), x.c, x.h, x.w, left, top, out->h, out->w, out->d());
    return out;
  }

  TensorPtr crop(const TensorPtr& xp, int left, int top, int width, int height) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, height, width);
    JKP_LAUNCH(k_crop, out->size(), x.d(), x.c, x.h, x.w, left, top, height, width, out->d());
    return out;
  }

  TensorPtr conv2d(const TensorPtr& xp, const Param& weight, const Param* bias, Act act) override {
    use();
    const CudaTensor& x = dev(xp);
    const int cout = weight.shape[0], cin = weight.shape[1], k = weight.shape[2];
    if (cin != x.c || weight.shape[3] != k || k % 2 == 0) throw std::runtime_error("conv2d: shape mismatch");
    CudaPtr out = make(cout, x.h, x.w);
    JKP_LAUNCH(k_conv2d, out->size(), x.d(), param(weight), bias ? param(*bias) : nullptr, cin, cout, x.h, x.w, k,
               act_id(act), out->d());
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
    Buffer<float> col(pool_, static_cast<size_t>(count(static_cast<long long>(cin) * 9 * hw)));
    JKP_LAUNCH(k_deform_sample, 9 * hw, x.d(), dev(offset).d(), cin, h, w, max_offset, col.get());
    CudaPtr out = make(cout, h, w);
    JKP_LAUNCH(k_deform_sum, out->size(), col.get(), param(weight), param(bias), cin * 9, cout, hw, act_id(act),
               out->d());
    return out;
  }

  TensorPtr add(const TensorPtr& ap, const TensorPtr& bp, Act act) override {
    use();
    const CudaTensor &a = dev(ap), &b = dev(bp);
    if (a.size() != b.size()) throw std::runtime_error("add: shape mismatch");
    CudaPtr out = make(a.c, a.h, a.w);
    JKP_LAUNCH(k_add, out->size(), a.d(), b.d(), out->size(), act_id(act), out->d());
    return out;
  }

  TensorPtr avg_pool(const TensorPtr& xp, int k) override {
    use();
    const CudaTensor& x = dev(xp);
    if (k < 1 || x.h % k || x.w % k) throw std::runtime_error("avg_pool: size is not a multiple of the window");
    CudaPtr out = make(x.c, x.h / k, x.w / k);
    JKP_LAUNCH(k_avg_pool, out->size(), x.d(), x.c, x.w, out->h, out->w, k, out->d());
    return out;
  }

  TensorPtr upsample(const TensorPtr& xp, int scale) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h * scale, x.w * scale);
    auto ratio = [](int in, int outn) {
      return outn > 1 ? static_cast<float>(in - 1) / static_cast<float>(outn - 1) : 0.0f;
    };
    JKP_LAUNCH(k_upsample, out->size(), x.d(), x.c, x.h, x.w, out->h, out->w, ratio(x.w, out->w), ratio(x.h, out->h),
               out->d());
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
        JKP_CUDA_CHECK(cudaMemcpy(out->d() + pos, x.d(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
      pos += x.size();
    }
    return out;
  }

  TensorPtr l2_normalize(const TensorPtr& xp) override {
    use();
    const CudaTensor& x = dev(xp);
    CudaPtr out = make(x.c, x.h, x.w);
    JKP_LAUNCH(k_l2_normalize, x.h * x.w, x.d(), x.c, x.h * x.w, out->d());
    return out;
  }

  void detect(const TensorPtr& score_map, const DetectParams& params, Keypoints* out) override {
    use();
    const CudaTensor& sm = dev(score_map);
    const int w = sm.w, h = sm.h, n = w * h;
    const float* s = sm.d();
    out->xy_norm.clear();
    out->score.clear();
    if (n == 0) return;

    // Non-maximum suppression (simple_nms from SuperGlue, two refinement rounds).
    Buffer<float> nms(pool_, n);
    {
      Buffer<float> tmp(pool_, n), mask(pool_, n), pooled_mask(pool_, n), supp(pool_, n), pooled_supp(pool_, n);
      auto max_filter = [&](const float* src, float* dst) {
        JKP_LAUNCH(k_max_filter, n, src, w, h, kRadius, 0, tmp.get());
        JKP_LAUNCH(k_max_filter, n, tmp.get(), w, h, kRadius, 1, dst);
      };
      max_filter(s, pooled_supp.get());
      JKP_LAUNCH(k_nms_init, n, s, pooled_supp.get(), n, mask.get());
      for (int round = 0; round < 2; ++round) {
        max_filter(mask.get(), pooled_mask.get());
        JKP_LAUNCH(k_nms_suppress, n, s, pooled_mask.get(), n, supp.get());
        max_filter(supp.get(), pooled_supp.get());
        JKP_LAUNCH(k_nms_update, n, supp.get(), pooled_supp.get(), pooled_mask.get(), n, mask.get());
      }
      JKP_LAUNCH(k_nms_final, n, s, mask.get(), w, h, kRadius, nms.get());
    }

    // Selection, on the device. `idx` receives the pixel indices of the keypoints in output order.
    const int blocks = (n + kScanBlock - 1) / kScanBlock;
    Buffer<unsigned> offsets(pool_, static_cast<size_t>(blocks) + 1);
    // Counts {nms `mode` th} and leaves the per-block offsets in `offsets`.
    auto select_count = [&](int mode, float th) {
      JKP_LAUNCH(k_select_count, blocks, nms.get(), n, mode, th, offsets.get());
      JKP_LAUNCH(k_select_scan, 1, offsets.get(), blocks);
      unsigned total = 0;
      JKP_CUDA_CHECK(cudaMemcpy(&total, offsets.get() + blocks, sizeof(total), cudaMemcpyDeviceToHost));
      return static_cast<int>(total);
    };
    // Writes the first `limit` of that set in raster order; must follow the matching select_count.
    auto select_write = [&](int mode, float th, int limit, int* dst) {
      JKP_LAUNCH(k_select_write, blocks, nms.get(), n, mode, th, offsets.get(), static_cast<unsigned>(limit), dst);
    };

    int kept = 0;         // number of keypoints
    int best = -1;        // >= 0: keep the `best` highest scores, best first
    float threshold = 0;  // otherwise: keep nms > threshold, in raster order
    if (params.top_k > 0) {
      best = std::min(params.top_k, n);
    } else {
      int passing = params.scores_th > 0 ? select_count(kGreater, threshold = params.scores_th) : 0;
      if (passing == 0) {  // fall back to the mean score
        Buffer<float> sum(pool_, 1);
        JKP_LAUNCH(k_sum_sequential, 1, s, n, sum.get());
        float total = 0;
        sum.download(&total, 1);
        passing = select_count(kGreater, threshold = total / static_cast<float>(n));
      }
      if (passing > params.n_limit) best = std::max(0, params.n_limit);
      else kept = passing;
    }
    if (best >= 0) kept = best;
    if (kept == 0) return;

    Buffer<int> idx(pool_, kept);
    if (best < 0) {
      select_write(kGreater, threshold, kept, idx.get());
    } else {
      // Radix select for the value of the `best`-th highest score, one byte of the float bits per pass.
      Buffer<unsigned> hist_dev(pool_, 256);
      unsigned hist[256], prefix = 0, mask = 0;
      int remaining = best;  // how many are still needed from the values matching the prefix
      for (int shift = 24; shift >= 0; shift -= 8) {
        JKP_CUDA_CHECK(cudaMemset(hist_dev.get(), 0, sizeof(hist)));
        JKP_LAUNCH(k_histogram, n, nms.get(), n, mask, prefix, shift, hist_dev.get());
        hist_dev.download(hist, 256);
        int bin = 255;
        for (; bin > 0 && static_cast<long long>(hist[bin]) < remaining; --bin) remaining -= static_cast<int>(hist[bin]);
        prefix |= static_cast<unsigned>(bin) << shift;
        mask |= 255u << shift;
      }
      float kth;
      std::memcpy(&kth, &prefix, sizeof(kth));
      // Everything above that value, then the first `remaining` equal to it in raster order.
      const int above = best - remaining;
      Buffer<int> unsorted(pool_, kept);
      if (select_count(kGreater, kth) != above) throw std::runtime_error("cuda: top-k selection is inconsistent");
      select_write(kGreater, kth, above, unsorted.get());
      select_count(kEqual, kth);
      select_write(kEqual, kth, remaining, unsorted.get() + above);
      JKP_LAUNCH(k_rank_sort, kept, nms.get(), unsorted.get(), kept, idx.get());
    }

    // Refinement and readback of the kept keypoints only.
    Buffer<float> xy(pool_, 2 * static_cast<size_t>(kept)), score(pool_, kept);
    JKP_LAUNCH(k_refine, kept, s, w, h, idx.get(), kept, xy.get(), score.get());
    out->xy_norm.resize(2 * static_cast<size_t>(kept));
    out->score.resize(kept);
    xy.download(out->xy_norm.data(), out->xy_norm.size());
    score.download(out->score.data(), out->score.size());
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

    // In chunks, so the per-keypoint work buffers (M*C floats each) stay small for any keypoint count.
    const int chunk = 8192;
    for (int begin = 0; begin < total; begin += chunk) {
      const int n = std::min(chunk, total - begin);
      const size_t samples = static_cast<size_t>(n) * M * c;
      Buffer<float> xy(pool_, 2 * static_cast<size_t>(n)), o0(pool_, static_cast<size_t>(n) * 2 * M),
          o1(pool_, static_cast<size_t>(n) * 2 * M), sampled(pool_, samples), feat(pool_, samples),
          d(pool_, static_cast<size_t>(n) * dim);
      xy.upload(kpts.xy_norm.data() + 2 * static_cast<size_t>(begin), xy.size());
      JKP_LAUNCH(k_sddh_offset0, n * 2 * M, f.d(), c, h, w, xy.get(), n, K, param(*p.offset0_w), param(*p.offset0_b),
                 2 * M, o0.get());
      JKP_LAUNCH(k_sddh_offset1, n * 2 * M, o0.get(), n, 2 * M, param(*p.offset1_w), param(*p.offset1_b), max_offset,
                 o1.get());
      JKP_LAUNCH(k_sddh_sample, n * M, f.d(), c, h, w, xy.get(), o1.get(), n, M, sampled.get());
      JKP_LAUNCH(k_sddh_sf, samples, sampled.get(), static_cast<int>(samples), c, param(*p.sf_w), feat.get());
      JKP_LAUNCH(k_sddh_aggregate, n * dim, feat.get(), n, M * c, dim, param(*p.agg_w), d.get());
      JKP_LAUNCH(k_normalize_rows, n, d.get(), n, dim);
      d.download(desc->data() + static_cast<size_t>(begin) * dim, d.size());
    }
  }

 private:
  // The current device is per host thread; select ours on entry to every operation.
  void use() const { JKP_CUDA_CHECK(cudaSetDevice(pool_->device())); }

  CudaPtr make(int c, int h, int w) { return std::make_shared<CudaTensor>(pool_, c, h, w); }

  // Device copy of a weight tensor, uploaded on first use. Weights outlive the backend (see jkp_ctx).
  const float* param(const Param& p) {
    auto it = params_.find(p.data);
    if (it != params_.end()) return it->second;
    float* d = nullptr;
    JKP_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(p.size(), 1) * sizeof(float)));
    params_[p.data] = d;
    JKP_CUDA_CHECK(cudaMemcpy(d, p.data, p.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
  }

  PoolPtr pool_;
  std::map<const float*, float*> params_;
};

}  // namespace

std::unique_ptr<Ops> make_cuda_ops(int device) { return std::make_unique<CudaOps>(device); }

}  // namespace jkp
