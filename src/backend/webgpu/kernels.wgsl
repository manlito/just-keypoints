// WebGPU backend kernels (WGSL): a kernel-by-kernel port of src/backend/metal/kernels.metal, same names,
// same arithmetic and order of operations per output value, CHW layout.
//
// Layout of this file (read by webgpu_ops.cpp):
// - Everything before the first `//@kernel` line is the prelude, included in every kernel's module.
// - `//@kernel NAME` starts a kernel section; `//@helpers NAME` starts a block that a kernel pulls in with a
//   `//@use NAME` line. Each kernel is compiled as its own module (prelude + helpers + section), so
//   sections declare their own bindings and may reuse names and binding numbers.
// - A section defines `fn run(i: i32)`; the host appends the entry point, which calls it once per thread
//   (1-D index over a 2-D grid of 256-thread groups; surplus threads have i >= the kernel's count).
// - `$T`, `$T4` (= T / 4), `$K`, `$O` are replaced per instance (WGSL has no templates): k_conv2d_t8_k3
//   is the section k_conv2d_tile with T = 8, K = 3; a `_f32` suffix sets O = f32 (the score map).
// - Binding 0 is the kernel's scalar arguments (`P`, 64 words, read with pi / pu / pf by position, in
//   the order the host passes them). Buffers bind to 1, 2, ... in the order the host passes them.
//
// `real` is the storage type of feature maps and of the dense network's weights: f32, or f16 in FP16
// mode (the host prepends `enable f16;` and the alias). Arithmetic is f32 in both modes.
//
// Differences from the Metal source forced by WGSL: no pointers into storage buffers (planes are a base
// index), no 8-bit storage (the image is read from 32-bit words), no infinity literal, and no way to turn
// off contraction into fused multiply-add (the WebGPU implementation decides).

struct Params {
  v: array<vec4<u32>, 16>,
}
@group(0) @binding(0) var<uniform> P: Params;

fn pu(i: u32) -> u32 { return P.v[i >> 2u][i & 3u]; }
fn pi(i: u32) -> i32 { return bitcast<i32>(pu(i)); }
fn pf(i: u32) -> f32 { return bitcast<f32>(pu(i)); }

const kActNone = 0;
const kActSelu = 1;
const kActSigmoid = 2;
const kActLeakyRelu = 3;  // slope 0.01
const kActRelu = 4;
const kGreater = 0;
const kRadius = 2;        // NMS and soft-argmax window radius
const kScanBlock = 256;   // elements handled by one thread in the compaction kernels
const kLowest = -3.4028234663852886e38;

fn selu(x: f32) -> f32 {
  let alpha = 1.6732632423543772848170429916717;
  let scale = 1.0507009873554804934193349852946;
  if (x > 0.0) { return scale * x; }
  return scale * (alpha * (exp(x) - 1.0));
}

fn activate(v: f32, act: i32) -> f32 {
  if (act == kActSelu) { return selu(v); }
  if (act == kActSigmoid) { return 1.0 / (1.0 + exp(-v)); }
  if (act == kActLeakyRelu) { return select(0.01 * v, v, v > 0.0); }
  if (act == kActRelu) { return max(v, 0.0); }
  return v;
}

// a / b rounded to nearest, as the CPU backend computes it. WGSL's division may be approximate (it is on
// Dawn's Metal path); two corrections with exact fused multiply-adds round it correctly.
// Used where the descriptor head truncates the result: a keypoint on an integer pixel must stay there.
fn div_rn(a: f32, b: f32) -> f32 {
  let y = 1.0 / b;
  var q = a * y;
  q = fma(fma(-q, b, a), y, q);
  return fma(fma(-q, b, a), y, q);
}

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
  if (v < lo) { return lo; }
  if (hi < v) { return hi; }
  return v;
}

// Bilinear sample with zeros outside the map (PyTorch grid_sample, padding_mode = zeros): four indices
// into a plane and their weights; corners outside the map get index 0 and weight 0.
struct Bilinear {
  idx: array<i32, 4>,
  wt: array<f32, 4>,
}

fn bilinear_setup(x: f32, y: f32, w: i32, h: i32) -> Bilinear {
  let x0f = floor(x);
  let y0f = floor(y);
  let x0 = i32(x0f);
  let y0 = i32(y0f);
  let wx1 = x - x0f;
  let wy1 = y - y0f;
  let wx0 = (x0f + 1.0) - x;
  let wy0 = (y0f + 1.0) - y;
  let xs = array<i32, 4>(x0, x0 + 1, x0, x0 + 1);
  let ys = array<i32, 4>(y0, y0, y0 + 1, y0 + 1);
  let ws = array<f32, 4>(wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1);
  var b: Bilinear;
  for (var k = 0; k < 4; k++) {
    let inside = xs[k] >= 0 && xs[k] < w && ys[k] >= 0 && ys[k] < h;
    b.idx[k] = select(0, ys[k] * w + xs[k], inside);
    b.wt[k] = select(0.0, ws[k], inside);
  }
  return b;
}

fn bilinear_zero() -> Bilinear {
  var b: Bilinear;
  for (var k = 0; k < 4; k++) {
    b.idx[k] = 0;
    b.wt[k] = 0.0;
  }
  return b;
}

// ---------------------------------------------------------------------------------------------
// Dense kernels. One thread per output element unless noted.
// ---------------------------------------------------------------------------------------------

//@kernel k_image
// 8-bit pixels, read from 32-bit words.
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<real>;
fn run(i: i32) {
  let stride = pi(0u);
  let channels = pi(1u);
  let w = pi(2u);
  let h = pi(3u);
  let hw = h * w;
  if (i >= 3 * hw) { return; }
  let c = i / hw;
  let y = (i % hw) / w;
  let x = i % w;
  let src_c = select(c, 0, channels == 1);
  let b = u32(y * stride + x * channels + src_c);
  let v = (src[b >> 2u] >> ((b & 3u) * 8u)) & 255u;
  dst[i] = real(f32(v) / 255.0);
}

//@kernel k_pad_replicate
@group(0) @binding(1) var<storage, read> src: array<real>;
@group(0) @binding(2) var<storage, read_write> dst: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let left = pi(3u);
  let top = pi(4u);
  let oh = pi(5u);
  let ow = pi(6u);
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let x = i % ow;
  let sy = min(max(y - top, 0), h - 1);
  let sx = min(max(x - left, 0), w - 1);
  dst[i] = src[(ch * h + sy) * w + sx];
}

//@kernel k_crop
// $O: real, or f32 for the score map (k_crop_f32).
@group(0) @binding(1) var<storage, read> src: array<$O>;
@group(0) @binding(2) var<storage, read_write> dst: array<$O>;
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let left = pi(3u);
  let top = pi(4u);
  let oh = pi(5u);
  let ow = pi(6u);
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let x = i % ow;
  dst[i] = src[(ch * h + y + top) * w + x + left];
}

//@kernel k_conv2d
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> weight: array<real>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let has_bias = pi(0u);
  let cin = pi(1u);
  let cout = pi(2u);
  let h = pi(3u);
  let w = pi(4u);
  let k = pi(5u);
  let act = pi(6u);
  let hw = h * w;
  if (i >= cout * hw) { return; }
  let oc = i / hw;
  let y = (i % hw) / w;
  let xo = i % w;
  let pad = k / 2;
  var acc = 0.0;
  if (has_bias != 0) { acc = f32(bias[oc]); }
  var wk = oc * cin * k * k;
  for (var ic = 0; ic < cin; ic++) {
    let base = ic * hw;
    for (var ky = 0; ky < k; ky++) {
      let sy = y + ky - pad;
      if (sy < 0 || sy >= h) { continue; }
      for (var kx = 0; kx < k; kx++) {
        let sx = xo + kx - pad;
        if (sx < 0 || sx >= w) { continue; }
        acc += f32(weight[wk + ky * k + kx]) * f32(x[base + sy * w + sx]);
      }
    }
    wk += k * k;
  }
  out[i] = real(activate(acc, act));
}

//@kernel k_conv2d_tile
// T output channels per thread for one pixel, KxK kernel. Weights tap-major: w[tap * cpad + oc],
// tap = (ic * K + ky) * K + kx, cpad = cout rounded up to T, read four at a time.
// Grid: ceil(cout / T) * h * w, pixel fastest. Each output accumulates in k_conv2d's order.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<$O>;
fn run(i: i32) {
  let has_bias = pi(0u);
  let cin = pi(1u);
  let cout = pi(2u);
  let h = pi(3u);
  let w = pi(4u);
  let act = pi(5u);
  let hw = h * w;
  let blocks = (cout + $T - 1) / $T;
  if (i >= blocks * hw) { return; }
  let ob = i / hw;
  let pix = i % hw;
  let y = pix / w;
  let xo = pix % w;
  let pad = $K / 2;
  let oc0 = ob * $T;
  var acc: array<vec4<f32>, $T4>;
  for (var t = 0; t < $T; t++) {
    var b = 0.0;
    if (has_bias != 0 && oc0 + t < cout) { b = f32(bias[oc0 + t]); }
    acc[t / 4][t % 4] = b;
  }
  for (var ic = 0; ic < cin; ic++) {
    let base = ic * hw;
    let wic = ic * $K * $K;
    for (var ky = 0; ky < $K; ky++) {
      let sy = y + ky - pad;
      if (sy < 0 || sy >= h) { continue; }
      for (var kx = 0; kx < $K; kx++) {
        let sx = xo + kx - pad;
        if (sx < 0 || sx >= w) { continue; }
        let v = vec4<f32>(f32(x[base + sy * w + sx]));
        let wp = ((wic + ky * $K + kx) * blocks + ob) * $T4;
        for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], vec4<f32>(weight[wp + q]), v); }
      }
    }
  }
  for (var t = 0; t < $T; t++) {
    if (oc0 + t < cout) { out[(oc0 + t) * hw + pix] = $O(activate(acc[t / 4][t % 4], act)); }
  }
}

//@kernel k_deform_sample
// One thread per (kernel tap, pixel): samples every input channel at the offset position.
// col[(ic * 9 + tap) * hw + pixel].
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> off: array<real>;
@group(0) @binding(3) var<storage, read_write> col: array<real>;
fn run(i: i32) {
  let cin = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let max_offset = pf(3u);
  let hw = h * w;
  if (i >= 9 * hw) { return; }
  let k = i / hw;
  let pix = i % hw;
  let y = pix / w;
  let xo = pix % w;
  let off_y = f32(off[2 * k * hw + pix]);
  let off_x = f32(off[(2 * k + 1) * hw + pix]);
  let py = f32(y - 1 + k / 3) + clampf(off_y, -max_offset, max_offset);
  let px = f32(xo - 1 + k % 3) + clampf(off_x, -max_offset, max_offset);
  var b = bilinear_zero();
  if (py > -1.0 && py < f32(h) && px > -1.0 && px < f32(w)) { b = bilinear_setup(px, py, w, h); }
  for (var ic = 0; ic < cin; ic++) {
    let base = ic * hw;
    let v = b.wt[0] * f32(x[base + b.idx[0]]) + b.wt[1] * f32(x[base + b.idx[1]]) +
            b.wt[2] * f32(x[base + b.idx[2]]) + b.wt[3] * f32(x[base + b.idx[3]]);
    col[(ic * 9 + k) * hw + pix] = real(v);
  }
}

//@kernel k_deform_sum_tile
// Weighted sum over the sampled columns, T output channels per thread; weights tap-major,
// tap = ic * 9 + kernel tap.
@group(0) @binding(1) var<storage, read> col: array<real>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let taps = pi(0u);
  let cout = pi(1u);
  let hw = pi(2u);
  let act = pi(3u);
  let blocks = (cout + $T - 1) / $T;
  if (i >= blocks * hw) { return; }
  let ob = i / hw;
  let pix = i % hw;
  let oc0 = ob * $T;
  var acc: array<vec4<f32>, $T4>;
  for (var t = 0; t < $T; t++) {
    var b = 0.0;
    if (oc0 + t < cout) { b = f32(bias[oc0 + t]); }
    acc[t / 4][t % 4] = b;
  }
  for (var j = 0; j < taps; j++) {
    let v = vec4<f32>(f32(col[j * hw + pix]));
    let wp = (j * blocks + ob) * $T4;
    for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], vec4<f32>(weight[wp + q]), v); }
  }
  for (var t = 0; t < $T; t++) {
    if (oc0 + t < cout) { out[(oc0 + t) * hw + pix] = real(activate(acc[t / 4][t % 4], act)); }
  }
}

//@kernel k_add
@group(0) @binding(1) var<storage, read> a: array<real>;
@group(0) @binding(2) var<storage, read> b: array<real>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let n = pi(0u);
  let act = pi(1u);
  if (i >= n) { return; }
  out[i] = real(activate(f32(a[i]) + f32(b[i]), act));
}

//@kernel k_avg_pool
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let w = pi(1u);
  let oh = pi(2u);
  let ow = pi(3u);
  let k = pi(4u);
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let xo = i % ow;
  let base = ch * (oh * k) * w;
  var sum = 0.0;
  for (var j = 0; j < k; j++) {
    for (var q = 0; q < k; q++) { sum += f32(x[base + (y * k + j) * w + xo * k + q]); }
  }
  out[i] = real(sum / f32(k * k));
}

//@kernel k_upsample
// Bilinear, align_corners = true: source position = dst * (in - 1) / (out - 1); sx, sy are those ratios.
// Writes at element offset dst_off of `out` (a part of a concatenation).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let oh = pi(3u);
  let ow = pi(4u);
  let sx = pf(5u);
  let sy = pf(6u);
  let dst_off = pi(7u);
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let xo = i % ow;
  let src_x = sx * f32(xo);
  let src_y = sy * f32(y);
  let x0 = min(i32(src_x), w - 1);
  let y0 = min(i32(src_y), h - 1);
  let x1 = x0 + select(0, 1, x0 < w - 1);
  let y1 = y0 + select(0, 1, y0 < h - 1);
  let w1 = src_x - f32(x0);
  let w0 = 1.0 - w1;
  let h1 = src_y - f32(y0);
  let h0 = 1.0 - h1;
  let r0 = (ch * h + y0) * w;
  let r1 = (ch * h + y1) * w;
  out[dst_off + i] = real(h0 * (w0 * f32(x[r0 + x0]) + w1 * f32(x[r0 + x1])) +
                          h1 * (w0 * f32(x[r1 + x0]) + w1 * f32(x[r1 + x1])));
}

//@kernel k_copy
@group(0) @binding(1) var<storage, read> src: array<real>;
@group(0) @binding(2) var<storage, read_write> dst: array<real>;
fn run(i: i32) {
  let n = pi(0u);
  let dst_off = pi(1u);
  if (i >= n) { return; }
  dst[dst_off + i] = src[i];
}

//@kernel k_l2_normalize
// One thread per pixel.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let hw = pi(1u);
  if (i >= hw) { return; }
  var norm = 0.0;
  for (var ch = 0; ch < c; ch++) {
    let v = f32(x[ch * hw + i]);
    norm += v * v;
  }
  norm = max(sqrt(norm), 1e-12);
  for (var ch = 0; ch < c; ch++) { out[ch * hw + i] = real(f32(x[ch * hw + i]) / norm); }
}

//@kernel k_l2_normalize_crop
// L2 normalization across channels of the cropped region of x (c x h x w) at (left, top); out is
// c x oh x ow. One thread per output pixel.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let left = pi(3u);
  let top = pi(4u);
  let oh = pi(5u);
  let ow = pi(6u);
  let ohw = oh * ow;
  if (i >= ohw) { return; }
  let hw = h * w;
  let src = (i / ow + top) * w + i % ow + left;
  var norm = 0.0;
  for (var ch = 0; ch < c; ch++) {
    let v = f32(x[ch * hw + src]);
    norm += v * v;
  }
  norm = max(sqrt(norm), 1e-12);
  for (var ch = 0; ch < c; ch++) { out[ch * ohw + i] = real(f32(x[ch * hw + src]) / norm); }
}

// ---------------------------------------------------------------------------------------------
// Pyramid: the concatenation of up to four maps, each upsampled to the full resolution (k_upsample's
// arithmetic, or as is for scale 1). These kernels compute its values where they are needed instead of
// storing it. Parameters 0-27 are PyramidArgs (webgpu_ops.cpp); a kernel's own scalars follow at 28.
// The four parts bind to 1-4 (an unused part binds part 0 again and has c = 0).
// ---------------------------------------------------------------------------------------------

//@helpers pyramid
@group(0) @binding(1) var<storage, read> p0: array<real>;
@group(0) @binding(2) var<storage, read> p1: array<real>;
@group(0) @binding(3) var<storage, read> p2: array<real>;
@group(0) @binding(4) var<storage, read> p3: array<real>;

fn pa_c(p: i32) -> i32 { return pi(u32(p)); }
fn pa_h(p: i32) -> i32 { return pi(4u + u32(p)); }
fn pa_w(p: i32) -> i32 { return pi(8u + u32(p)); }
fn pa_scale(p: i32) -> i32 { return pi(12u + u32(p)); }
fn pa_sx(p: i32) -> f32 { return pf(16u + u32(p)); }
fn pa_sy(p: i32) -> f32 { return pf(20u + u32(p)); }
fn pa_H() -> i32 { return pi(24u); }
fn pa_W() -> i32 { return pi(25u); }
fn pa_left() -> i32 { return pi(26u); }
fn pa_top() -> i32 { return pi(27u); }
fn pa_channels() -> i32 { return pa_c(0) + pa_c(1) + pa_c(2) + pa_c(3); }

fn part_load(p: i32, idx: i32) -> f32 {
  switch p {
    case 0: { return f32(p0[idx]); }
    case 1: { return f32(p1[idx]); }
    case 2: { return f32(p2[idx]); }
    default: { return f32(p3[idx]); }
  }
}

// Where one part is read for a full-resolution pixel.
struct Taps {
  i00: i32, i01: i32, i10: i32, i11: i32,
  w0: f32, w1: f32, h0: f32, h1: f32,
  direct: bool,
}

fn part_taps(p: i32, X: i32, Y: i32) -> Taps {
  var t: Taps;
  let w = pa_w(p);
  let h = pa_h(p);
  t.direct = pa_scale(p) == 1;
  if (t.direct) {
    t.i00 = Y * w + X;
    t.i01 = t.i00;
    t.i10 = t.i00;
    t.i11 = t.i00;
    t.w0 = 1.0;
    t.h0 = 1.0;
    t.w1 = 0.0;
    t.h1 = 0.0;
    return t;
  }
  let src_x = pa_sx(p) * f32(X);
  let src_y = pa_sy(p) * f32(Y);
  let x0 = min(i32(src_x), w - 1);
  let y0 = min(i32(src_y), h - 1);
  let x1 = x0 + select(0, 1, x0 < w - 1);
  let y1 = y0 + select(0, 1, y0 < h - 1);
  t.w1 = src_x - f32(x0);
  t.w0 = 1.0 - t.w1;
  t.h1 = src_y - f32(y0);
  t.h0 = 1.0 - t.h1;
  t.i00 = y0 * w + x0;
  t.i01 = y0 * w + x1;
  t.i10 = y1 * w + x0;
  t.i11 = y1 * w + x1;
  return t;
}

// Channel `ch` of part p at those taps.
fn part_value(p: i32, ch: i32, t: Taps) -> f32 {
  let plane = ch * (pa_h(p) * pa_w(p));
  if (t.direct) { return part_load(p, plane + t.i00); }
  return t.h0 * (t.w0 * part_load(p, plane + t.i00) + t.w1 * part_load(p, plane + t.i01)) +
         t.h1 * (t.w0 * part_load(p, plane + t.i10) + t.w1 * part_load(p, plane + t.i11));
}

// L2 norm across the channels of the pyramid at one pixel, as k_l2_normalize computes it.
fn pyramid_norm(X: i32, Y: i32) -> f32 {
  var norm = 0.0;
  for (var p = 0; p < 4; p++) {
    let t = part_taps(p, X, Y);
    for (var ch = 0; ch < pa_c(p); ch++) {
      let v = part_value(p, ch, t);
      norm += v * v;
    }
  }
  return max(sqrt(norm), 1e-12);
}

//@kernel k_conv1x1_pyramid
// 1x1 convolution on a pyramid, T output channels per thread; same accumulation order as
// k_conv2d_tile (K = 1) on the stored pyramid, tap-major weights. Grid: ceil(cout / T) * H * W.
//@use pyramid
@group(0) @binding(5) var<storage, read> weight: array<vec4<real>>;
@group(0) @binding(6) var<storage, read> bias: array<real>;
@group(0) @binding(7) var<storage, read_write> out: array<$O>;
fn run(i: i32) {
  let has_bias = pi(28u);
  let cout = pi(29u);
  let act = pi(30u);
  let hw = pa_H() * pa_W();
  let blocks = (cout + $T - 1) / $T;
  if (i >= blocks * hw) { return; }
  let ob = i / hw;
  let pix = i % hw;
  let Y = pix / pa_W();
  let X = pix % pa_W();
  let oc0 = ob * $T;
  var acc: array<vec4<f32>, $T4>;
  for (var t = 0; t < $T; t++) {
    var b = 0.0;
    if (has_bias != 0 && oc0 + t < cout) { b = f32(bias[oc0 + t]); }
    acc[t / 4][t % 4] = b;
  }
  var base = 0;
  for (var p = 0; p < 4; p++) {
    if (pa_c(p) == 0) { continue; }
    let taps = part_taps(p, X, Y);
    for (var ch = 0; ch < pa_c(p); ch++) {
      let v = vec4<f32>(part_value(p, ch, taps));
      let wp = ((base + ch) * blocks + ob) * $T4;
      for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], vec4<f32>(weight[wp + q]), v); }
    }
    base += pa_c(p);
  }
  for (var t = 0; t < $T; t++) {
    if (oc0 + t < cout) { out[(oc0 + t) * hw + pix] = $O(activate(acc[t / 4][t % 4], act)); }
  }
}

// ---------------------------------------------------------------------------------------------
// Detection kernels (float score map)
// ---------------------------------------------------------------------------------------------

//@kernel k_max_filter
// Separable (2r+1) max filter, window clipped at the borders. dir 0 = along x, 1 = along y.
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
fn run(i: i32) {
  let w = pi(0u);
  let h = pi(1u);
  let r = pi(2u);
  let dir = pi(3u);
  if (i >= w * h) { return; }
  let y = i / w;
  let x = i % w;
  var m = src[i];
  if (dir == 0) {
    for (var k = max(0, x - r); k <= min(w - 1, x + r); k++) { m = max(m, src[y * w + k]); }
  } else {
    for (var k = max(0, y - r); k <= min(h - 1, y + r); k++) { m = max(m, src[k * w + x]); }
  }
  dst[i] = m;
}

//@kernel k_nms_init
// mask = 1 where the score is the maximum of its window.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> pooled: array<f32>;
@group(0) @binding(3) var<storage, read_write> mask: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  mask[i] = select(0.0, 1.0, s[i] == pooled[i]);
}

//@kernel k_nms_suppress
// Scores with everything near a current maximum (pooled_mask > 0) set to zero.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> pooled_mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> supp: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  supp[i] = select(s[i], 0.0, pooled_mask[i] > 0.0);
}

//@kernel k_nms_update
// Adds the maxima of the suppressed scores to the mask.
@group(0) @binding(1) var<storage, read> supp: array<f32>;
@group(0) @binding(2) var<storage, read> pooled_supp: array<f32>;
@group(0) @binding(3) var<storage, read> pooled_mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> mask: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  let suppressed = pooled_mask[i] > 0.0;
  if (supp[i] == pooled_supp[i] && !suppressed) { mask[i] = 1.0; }
}

//@kernel k_nms_final
// nms = score at maxima away from the border, zero elsewhere.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> nms: array<f32>;
fn run(i: i32) {
  let w = pi(0u);
  let h = pi(1u);
  let r = pi(2u);
  if (i >= w * h) { return; }
  let y = i / w;
  let x = i % w;
  let border = x < r || y < r || x >= w - r || y >= h - r;
  nms[i] = select(0.0, s[i], mask[i] > 0.0 && !border);
}

//@kernel k_sum_sequential
// Sequential float32 sum in one thread: the mean-score fallback threshold must not depend on the
// summation order. Only runs when no score passes the configured threshold.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i != 0) { return; }
  var sum = 0.0;
  for (var k = 0; k < n; k++) { sum += v[k]; }
  out[0] = sum;
}

//@kernel k_select_count
// Ordered compaction of {i : selected(v[i])}, three steps (as on CUDA, all on the device).
// 1. One thread per block of kScanBlock elements counts the selected elements.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
fn run(b: i32) {
  let n = pi(0u);
  let mode = pi(1u);
  let th = pf(2u);
  let begin = b * kScanBlock;
  if (begin >= n) { return; }
  let end = min(n, begin + kScanBlock);
  var count = 0u;
  for (var i = begin; i < end; i++) {
    let sel = select(v[i] == th, v[i] > th, mode == kGreater);
    count += select(0u, 1u, sel);
  }
  counts[b] = count;
}

//@kernel k_select_scan
// 2. Exclusive prefix sum over the block counts, in place, in one thread; counts[blocks] gets the total.
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
fn run(i: i32) {
  let blocks = pi(0u);
  if (i != 0) { return; }
  var sum = 0u;
  for (var b = 0; b < blocks; b++) {
    let c = counts[b];
    counts[b] = sum;
    sum += c;
  }
  counts[blocks] = sum;
}

//@kernel k_select_write
// 3. One thread per block writes its indices at the block offset (plus out_off); only the first `limit`.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<i32>;
fn run(b: i32) {
  let n = pi(0u);
  let mode = pi(1u);
  let th = pf(2u);
  let limit = pu(3u);
  let out_off = pi(4u);
  let begin = b * kScanBlock;
  if (begin >= n) { return; }
  let end = min(n, begin + kScanBlock);
  var pos = offsets[b];
  for (var i = begin; i < end && pos < limit; i++) {
    let sel = select(v[i] == th, v[i] > th, mode == kGreater);
    if (sel) {
      out[out_off + i32(pos)] = i;
      pos++;
    }
  }
}

//@kernel k_fill_u32
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
fn run(i: i32) {
  let n = pi(0u);
  let value = pu(1u);
  if (i >= n) { return; }
  dst[i] = value;
}

//@kernel k_histogram
// Radix select: histogram of 16 bits of the float bits, over the values whose higher bits equal `prefix`.
// Values are non-negative, so the bit patterns order like the floats.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;
fn run(i: i32) {
  let n = pi(0u);
  let mask = pu(1u);
  let prefix = pu(2u);
  let shift = pu(3u);
  if (i >= n) { return; }
  let bits = bitcast<u32>(v[i]);
  if ((bits & mask) == prefix) { atomicAdd(&hist[(bits >> shift) & 65535u], 1u); }
}

//@kernel k_rank_sort
// Rank sort of k indices by (value descending, index ascending). One thread per element counts the
// elements that come before it; the keys are distinct, so every rank is used exactly once.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> sorted: array<i32>;
fn run(i: i32) {
  let k = pi(0u);
  if (i >= k) { return; }
  let id = idx[i];
  let vi = v[id];
  var rank = 0;
  for (var j = 0; j < k; j++) {
    let jd = idx[j];
    let vj = v[jd];
    rank += select(0, 1, vj > vi || (vj == vi && jd < id));
  }
  sorted[rank] = id;
}

//@kernel k_refine
// Soft-argmax refinement in a 5x5 window (temperature 0.1); outside the map counts as score 0.
// One thread per keypoint.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> xy_norm: array<f32>;
@group(0) @binding(4) var<storage, read_write> score: array<f32>;
fn run(i: i32) {
  let w = pi(0u);
  let h = pi(1u);
  let n = pi(2u);
  if (i >= n) { return; }
  let temperature = 0.1;
  let wh = array<f32, 2>(f32(w - 1), f32(h - 1));
  let x = idx[i] % w;
  let y = idx[i] / w;
  var patches: array<f32, 25>;
  var max_v = kLowest;
  for (var dy = -kRadius; dy <= kRadius; dy++) {
    for (var dx = -kRadius; dx <= kRadius; dx++) {
      let px = x + dx;
      let py = y + dy;
      var v = 0.0;
      if (px >= 0 && px < w && py >= 0 && py < h) { v = s[py * w + px]; }
      patches[(dy + kRadius) * 5 + dx + kRadius] = v;
      max_v = max(max_v, v);
    }
  }
  var sum = 0.0;
  var sx = 0.0;
  var sy = 0.0;
  for (var j = 0; j < 25; j++) {
    let e = exp((patches[j] - max_v) / temperature);
    sum += e;
    sx += e * f32(j % 5 - kRadius);
    sy += e * f32(j / 5 - kRadius);
  }
  // Normalized to [-1, 1]; the descriptor head truncates the pixel position recovered from this value,
  // so the divisions are rounded as on the CPU (div_rn).
  let xy = array<f32, 2>(f32(x) + div_rn(sx, sum), f32(y) + div_rn(sy, sum));
  let kn0 = div_rn(xy[0], wh[0]) * 2.0 - 1.0;
  let kn1 = div_rn(xy[1], wh[1]) * 2.0 - 1.0;
  xy_norm[2 * i] = kn0;
  xy_norm[2 * i + 1] = kn1;

  // Score: bilinear sample of the score map at the refined position.
  let gx = ((kn0 + 1.0) / 2.0) * wh[0];
  let gy = ((kn1 + 1.0) / 2.0) * wh[1];
  let b = bilinear_setup(gx, gy, w, h);
  score[i] = b.wt[0] * s[b.idx[0]] + b.wt[1] * s[b.idx[1]] + b.wt[2] * s[b.idx[2]] + b.wt[3] * s[b.idx[3]];
}

// ---------------------------------------------------------------------------------------------
// Descriptor head (SDDH) kernels. n keypoints, C channels, KxK patches, M sample positions.
// Only the feature map `f` is `real`; the head's weights and buffers are f32 in both modes.
// ---------------------------------------------------------------------------------------------

//@kernel k_sddh_offset0
// Offsets, first layer: KxK convolution on the patches centred on the truncated keypoint (zeros outside
// the map), then SELU. One thread per (keypoint, output channel); o0[n][2M].
@group(0) @binding(1) var<storage, read> f: array<real>;
@group(0) @binding(2) var<storage, read> xy_norm: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> o0: array<f32>;
fn run(t: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let n = pi(3u);
  let K = pi(4u);
  let m2 = pi(5u);
  if (t >= n * m2) { return; }
  let i = t / m2;
  let j = t % m2;
  let hw = h * w;
  let lo = (K - 1) / 2;
  let kx = (xy_norm[2 * i] / 2.0 + 0.5) * f32(w - 1);
  let ky = (xy_norm[2 * i + 1] / 2.0 + 0.5) * f32(h - 1);
  let ix = i32(kx);  // truncation
  let iy = i32(ky);
  let wk = j * c * K * K;
  var acc = 0.0;
  for (var ch = 0; ch < c; ch++) {
    let plane = ch * hw;
    for (var a = 0; a < K; a++) {
      for (var b = 0; b < K; b++) {
        let py = iy - lo + a;
        let px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          acc += weight[wk + (ch * K + a) * K + b] * f32(f[plane + py * w + px]);
        }
      }
    }
  }
  o0[t] = selu(acc + bias[j]);
}

//@kernel k_sddh_offset1
// Offsets, second layer: 1x1 convolution, clamped. o1[n][2M] (x offsets first, then y).
@group(0) @binding(1) var<storage, read> o0: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> o1: array<f32>;
fn run(t: i32) {
  let n = pi(0u);
  let m2 = pi(1u);
  let max_offset = pf(2u);
  if (t >= n * m2) { return; }
  let i = t / m2;
  let j = t % m2;
  let wk = j * m2;
  let inp = i * m2;
  var acc = 0.0;
  for (var q = 0; q < m2; q++) { acc += weight[wk + q] * o0[inp + q]; }
  o1[t] = clampf(acc + bias[j], -max_offset, max_offset);
}

//@kernel k_sddh_sample
// Bilinear samples of the feature map at keypoint + offset. One thread per (keypoint, position);
// sampled[n][M][C].
@group(0) @binding(1) var<storage, read> f: array<real>;
@group(0) @binding(2) var<storage, read> xy_norm: array<f32>;
@group(0) @binding(3) var<storage, read> o1: array<f32>;
@group(0) @binding(4) var<storage, read_write> sampled: array<f32>;
fn run(t: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let n = pi(3u);
  let M = pi(4u);
  if (t >= n * M) { return; }
  let i = t / M;
  let m = t % M;
  let hw = h * w;
  let wh = array<f32, 2>(f32(w - 1), f32(h - 1));
  let kx = (xy_norm[2 * i] / 2.0 + 0.5) * wh[0];
  let ky = (xy_norm[2 * i + 1] / 2.0 + 0.5) * wh[1];
  let off = i * 2 * M;
  var gx = 2.0 * (kx + o1[off + m]) / wh[0] - 1.0;
  var gy = 2.0 * (ky + o1[off + M + m]) / wh[1] - 1.0;
  gx = ((gx + 1.0) / 2.0) * wh[0];
  gy = ((gy + 1.0) / 2.0) * wh[1];
  let b = bilinear_setup(gx, gy, w, h);
  let out = t * c;
  for (var ch = 0; ch < c; ch++) {
    let plane = ch * hw;
    sampled[out + ch] = b.wt[0] * f32(f[plane + b.idx[0]]) + b.wt[1] * f32(f[plane + b.idx[1]]) +
                        b.wt[2] * f32(f[plane + b.idx[2]]) + b.wt[3] * f32(f[plane + b.idx[3]]);
  }
}

//@kernel k_sddh_sf_tile
// 1x1 convolution + SELU on the samples, T output channels per thread; weights tap-major (input channel
// major). One thread per (keypoint, position, block of T output channels).
@group(0) @binding(1) var<storage, read> sampled: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> feat: array<f32>;
fn run(i: i32) {
  let total = pi(0u);
  let c = pi(1u);
  let blocks = (c + $T - 1) / $T;
  if (i >= total / c * blocks) { return; }
  let s = i / blocks;
  let ob = i % blocks;
  let oc0 = ob * $T;
  let inp = s * c;
  var acc: array<vec4<f32>, $T4>;
  for (var q = 0; q < $T4; q++) { acc[q] = vec4<f32>(0.0); }
  for (var ch = 0; ch < c; ch++) {
    let v = vec4<f32>(sampled[inp + ch]);
    let wp = (ch * blocks + ob) * $T4;
    for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], weight[wp + q], v); }
  }
  for (var t = 0; t < $T; t++) {
    if (oc0 + t < c) { feat[s * c + oc0 + t] = selu(acc[t / 4][t % 4]); }
  }
}

//@kernel k_sddh_aggregate
// Aggregation with one CxD matrix per position. One thread per (keypoint, descriptor element).
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> desc: array<f32>;
fn run(t: i32) {
  let n = pi(0u);
  let mc = pi(1u);
  let dim = pi(2u);
  if (t >= n * dim) { return; }
  let i = t / dim;
  let j = t % dim;
  let inp = i * mc;
  var d = 0.0;
  for (var q = 0; q < mc; q++) { d += feat[inp + q] * weight[q * dim + j]; }
  desc[t] = d;
}

//@kernel k_sddh_aggregate_tile
// k_sddh_aggregate, T descriptor elements per thread (dim is a multiple of T); weight[M*C][dim] as it is.
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> desc: array<f32>;
fn run(t: i32) {
  let n = pi(0u);
  let mc = pi(1u);
  let dim = pi(2u);
  let blocks = dim / $T;
  if (t >= n * blocks) { return; }
  let i = t / blocks;
  let j0 = (t % blocks) * $T;
  let inp = i * mc;
  var acc: array<vec4<f32>, $T4>;
  for (var q = 0; q < $T4; q++) { acc[q] = vec4<f32>(0.0); }
  for (var k = 0; k < mc; k++) {
    let v = vec4<f32>(feat[inp + k]);
    let wp = (k * dim + j0) / 4;
    for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], v, weight[wp + q]); }
  }
  for (var q = 0; q < $T; q++) { desc[i * dim + j0 + q] = acc[q / 4][q % 4]; }
}

//@kernel k_normalize_rows
// L2 normalization of each descriptor. One thread per keypoint.
@group(0) @binding(1) var<storage, read_write> desc: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  let dim = pi(1u);
  if (i >= n) { return; }
  let d = i * dim;
  var norm = 0.0;
  for (var j = 0; j < dim; j++) { norm += desc[d + j] * desc[d + j]; }
  norm = max(sqrt(norm), 1e-12);
  for (var j = 0; j < dim; j++) { desc[d + j] = desc[d + j] / norm; }
}

// ---------------------------------------------------------------------------------------------
// Descriptor head on a pyramid: the feature map is the L2-normalized pyramid, cropped to w x h at
// (left, top). These kernels normalize only the pixels the keypoints touch; they replace k_sddh_offset0
// and k_sddh_sample and give the same values as those do on the stored feature map.
// ---------------------------------------------------------------------------------------------

//@kernel k_sddh_patch_pyramid
// Normalized KxK patches around the truncated keypoint. One thread per (keypoint, patches pixel);
// patches[n][K*K][C], left untouched (and unused) for pixels outside the map.
//@use pyramid
@group(0) @binding(5) var<storage, read> xy_norm: array<f32>;
@group(0) @binding(6) var<storage, read_write> patches: array<f32>;
fn run(t: i32) {
  let h = pi(28u);
  let w = pi(29u);
  let n = pi(30u);
  let K = pi(31u);
  let kk = K * K;
  if (t >= n * kk) { return; }
  let i = t / kk;
  let q = t % kk;
  let lo = (K - 1) / 2;
  let kx = (xy_norm[2 * i] / 2.0 + 0.5) * f32(w - 1);
  let ky = (xy_norm[2 * i + 1] / 2.0 + 0.5) * f32(h - 1);
  let px = i32(kx) - lo + q % K;  // truncation
  let py = i32(ky) - lo + q / K;
  if (px < 0 || px >= w || py < 0 || py >= h) { return; }
  let X = px + pa_left();
  let Y = py + pa_top();
  let norm = pyramid_norm(X, Y);
  var out = t * pa_channels();
  for (var p = 0; p < 4; p++) {
    let taps = part_taps(p, X, Y);
    for (var ch = 0; ch < pa_c(p); ch++) {
      patches[out] = part_value(p, ch, taps) / norm;
      out++;
    }
  }
}

//@kernel k_sddh_offset0_patch
// k_sddh_offset0 on the patches of k_sddh_patch_pyramid.
@group(0) @binding(1) var<storage, read> patches: array<f32>;
@group(0) @binding(2) var<storage, read> xy_norm: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> o0: array<f32>;
fn run(t: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let n = pi(3u);
  let K = pi(4u);
  let m2 = pi(5u);
  if (t >= n * m2) { return; }
  let i = t / m2;
  let j = t % m2;
  let lo = (K - 1) / 2;
  let kx = (xy_norm[2 * i] / 2.0 + 0.5) * f32(w - 1);
  let ky = (xy_norm[2 * i + 1] / 2.0 + 0.5) * f32(h - 1);
  let ix = i32(kx);  // truncation
  let iy = i32(ky);
  let wk = j * c * K * K;
  let inp = i * K * K * c;
  var acc = 0.0;
  for (var ch = 0; ch < c; ch++) {
    for (var a = 0; a < K; a++) {
      for (var b = 0; b < K; b++) {
        let py = iy - lo + a;
        let px = ix - lo + b;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          acc += weight[wk + (ch * K + a) * K + b] * patches[inp + (a * K + b) * c + ch];
        }
      }
    }
  }
  o0[t] = selu(acc + bias[j]);
}

//@kernel k_sddh_sample_pyramid
// k_sddh_sample on the pyramid: the four pixels around each sample position are normalized here.
// One thread per (keypoint, position); sampled[n][M][C], C <= 128.
//@use pyramid
@group(0) @binding(5) var<storage, read> xy_norm: array<f32>;
@group(0) @binding(6) var<storage, read> o1: array<f32>;
@group(0) @binding(7) var<storage, read_write> sampled: array<f32>;
const kMaxPyramidC = 128;
fn run(t: i32) {
  let h = pi(28u);
  let w = pi(29u);
  let n = pi(30u);
  let M = pi(31u);
  if (t >= n * M) { return; }
  let i = t / M;
  let m = t % M;
  let wh = array<f32, 2>(f32(w - 1), f32(h - 1));
  let kx = (xy_norm[2 * i] / 2.0 + 0.5) * wh[0];
  let ky = (xy_norm[2 * i + 1] / 2.0 + 0.5) * wh[1];
  let off = i * 2 * M;
  var gx = 2.0 * (kx + o1[off + m]) / wh[0] - 1.0;
  var gy = 2.0 * (ky + o1[off + M + m]) / wh[1] - 1.0;
  gx = ((gx + 1.0) / 2.0) * wh[0];
  gy = ((gy + 1.0) / 2.0) * wh[1];
  let b = bilinear_setup(gx, gy, w, h);
  let c = pa_channels();
  var vals: array<f32, kMaxPyramidC>;
  var acc: array<f32, kMaxPyramidC>;
  // A corner outside the map has weight 0 and index 0, as in bilinear_apply: it adds 0 * (pixel 0).
  for (var k = 0; k < 4; k++) {
    let X = b.idx[k] % w + pa_left();
    let Y = b.idx[k] / w + pa_top();
    var pos = 0;
    var norm = 0.0;
    for (var p = 0; p < 4; p++) {
      let taps = part_taps(p, X, Y);
      for (var ch = 0; ch < pa_c(p); ch++) {
        let v = part_value(p, ch, taps);
        vals[pos] = v;
        pos++;
        norm += v * v;
      }
    }
    norm = max(sqrt(norm), 1e-12);
    for (var ch = 0; ch < c; ch++) {
      let v = b.wt[k] * (vals[ch] / norm);
      acc[ch] = select(acc[ch] + v, v, k == 0);
    }
  }
  let out = t * c;
  for (var ch = 0; ch < c; ch++) { sampled[out + ch] = acc[ch]; }
}

// ---------------------------------------------------------------------------------------------
// RaCo kernels (normalize, pad_reflect, detect_raco), ported from the Metal backend. Same float32 steps
// as the CPU backend, except that the softmax sum is a compensated float sum in blocks instead of a double
// sum in raster order, and softplus is computed without log1p (WGSL has none).
// ---------------------------------------------------------------------------------------------

//@kernel k_normalize
// (x - mean[c]) / std[c]; ms holds the c means, then the c standard deviations.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> ms: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let hw = pi(1u);
  if (i >= c * hw) { return; }
  let ch = i / hw;
  out[i] = real((f32(x[i]) - ms[ch]) / ms[c + ch]);
}

//@kernel k_pad_reflect
// Mirror padding by p on every side, without repeating the border pixel (PyTorch "reflect").
@group(0) @binding(1) var<storage, read> src: array<real>;
@group(0) @binding(2) var<storage, read_write> dst: array<real>;
fn mirror(i: i32, n: i32) -> i32 {
  if (i < 0) { return -i; }
  if (i >= n) { return 2 * (n - 1) - i; }
  return i;
}
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let p = pi(3u);
  let oh = h + 2 * p;
  let ow = w + 2 * p;
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let x = i % ow;
  dst[i] = src[(ch * h + mirror(y - p, h)) * w + mirror(x - p, w)];
}

//@kernel k_softmax_block_max
// Softmax over the whole map, in five steps: block maxima, the maximum (stats[0]), block sums of
// exp(l - max), the sum (stats[1]); then the probabilities. One thread per block of kScanBlock elements
// for the block steps, one thread for the reductions.
@group(0) @binding(1) var<storage, read> l: array<real>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
fn run(b: i32) {
  let n = pi(0u);
  let begin = b * kScanBlock;
  if (begin >= n) { return; }
  let end = min(n, begin + kScanBlock);
  var m = f32(l[begin]);
  for (var i = begin + 1; i < end; i++) { m = max(m, f32(l[i])); }
  partial[b] = m;
}

//@kernel k_softmax_max
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
fn run(i: i32) {
  let blocks = pi(0u);
  if (i != 0) { return; }
  var m = partial[0];
  for (var b = 1; b < blocks; b++) { m = max(m, partial[b]); }
  stats[0] = m;
}

//@kernel k_softmax_block_sum
// Kahan summation, so the result stays close to the CPU backend's double sum.
@group(0) @binding(1) var<storage, read> l: array<real>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
fn run(b: i32) {
  let n = pi(0u);
  let begin = b * kScanBlock;
  if (begin >= n) { return; }
  let end = min(n, begin + kScanBlock);
  let m = stats[0];
  var sum = 0.0;
  var comp = 0.0;
  for (var i = begin; i < end; i++) {
    let y = exp(f32(l[i]) - m) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  partial[b] = sum;
}

//@kernel k_softmax_sum
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
fn run(i: i32) {
  let blocks = pi(0u);
  if (i != 0) { return; }
  var sum = 0.0;
  var comp = 0.0;
  for (var b = 0; b < blocks; b++) {
    let y = partial[b] - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  stats[1] = sum;
}

//@kernel k_softmax_prob
@group(0) @binding(1) var<storage, read> l: array<real>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> prob: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  prob[i] = exp(f32(l[i]) - stats[0]) / stats[1];
}

//@kernel k_raco_nms
// nms = probability where it equals the maximum of its window, zero elsewhere (no border exclusion).
// Probabilities below the smallest normal float are not maxima, as on the CPU: the GPU may flush them to zero.
@group(0) @binding(1) var<storage, read> prob: array<f32>;
@group(0) @binding(2) var<storage, read> pooled: array<f32>;
@group(0) @binding(3) var<storage, read_write> nms: array<f32>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  let p = prob[i];
  nms[i] = select(0.0, p, p == pooled[i] && p >= 1.17549435e-38);
}

//@kernel k_raco_refine
// Subpixel refinement and sampling, one thread per keypoint: softmax of logit / temperature in the k x k
// window (outside the map counts as logit 0), expected offset; then bilinear samples at the refined
// position, clamped to the map, of the probabilities, the ranker map and the Cholesky map (softplus on
// channels 0 and 2 at each of the four neighbours).
// One output buffer (WebGPU guarantees only eight storage buffers per kernel): xy [2n], score [n], then
// rank [n] with a ranker map, then the Cholesky elements [3n] with a Cholesky map.
@group(0) @binding(1) var<storage, read> logits: array<real>;
@group(0) @binding(2) var<storage, read> prob: array<f32>;
@group(0) @binding(3) var<storage, read> ranker: array<real>;
@group(0) @binding(4) var<storage, read> chol: array<real>;
@group(0) @binding(5) var<storage, read> idx: array<i32>;
@group(0) @binding(6) var<storage, read_write> res: array<f32>;
// log(1 + exp(x)), linear above 20 as in PyTorch; accurate to a few ulp without log1p.
fn softplus(x: f32) -> f32 {
  if (x > 20.0) { return x; }
  let u = exp(x);
  let y = 1.0 + u;
  let z = y - 1.0;
  if (z == 0.0) { return u; }
  return log(y) * (u / z);
}
fn run(i: i32) {
  let has_ranker = pi(0u);
  let has_chol = pi(1u);
  let w = pi(2u);
  let h = pi(3u);
  let n = pi(4u);
  let k = pi(5u);
  let temperature = pf(6u);
  if (i >= n) { return; }
  let r = k / 2;
  let x = idx[i] % w;
  let y = idx[i] / w;
  var win: array<f32, 81>;
  var max_v = kLowest;
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let px = x + dx;
      let py = y + dy;
      var v = 0.0;
      if (px >= 0 && px < w && py >= 0 && py < h) { v = f32(logits[py * w + px]); }
      v = v / temperature;
      win[(dy + r) * k + dx + r] = v;
      max_v = max(max_v, v);
    }
  }
  var psum = 0.0;
  var sx = 0.0;
  var sy = 0.0;
  for (var j = 0; j < k * k; j++) {
    win[j] = exp(win[j] - max_v);
    psum += win[j];
  }
  for (var j = 0; j < k * k; j++) {
    let p = win[j] / psum;
    sx += p * f32(j % k - r);
    sy += p * f32(j / k - r);
  }
  let xy = array<f32, 2>(f32(x) + sx, f32(y) + sy);
  res[2 * i] = xy[0];
  res[2 * i + 1] = xy[1];

  // grid_sample, padding_mode = border, align_corners = true: normalize to [-1, 1], map back, clamp.
  let wh = array<f32, 2>(f32(w - 1), f32(h - 1));
  var g: array<f32, 2>;
  for (var a = 0; a < 2; a++) {
    var v = 0.0;
    if (wh[a] > 0.0) { v = ((2.0 * xy[a] / wh[a] - 1.0) + 1.0) / 2.0 * wh[a]; }
    g[a] = min(wh[a], max(v, 0.0));
  }
  let b = bilinear_setup(g[0], g[1], w, h);
  res[2 * n + i] = b.wt[0] * prob[b.idx[0]] + b.wt[1] * prob[b.idx[1]] + b.wt[2] * prob[b.idx[2]] +
                   b.wt[3] * prob[b.idx[3]];
  var at = 3 * n;
  if (has_ranker != 0) {
    res[at + i] = b.wt[0] * f32(ranker[b.idx[0]]) + b.wt[1] * f32(ranker[b.idx[1]]) +
                  b.wt[2] * f32(ranker[b.idx[2]]) + b.wt[3] * f32(ranker[b.idx[3]]);
    at += n;
  }
  if (has_chol != 0) {
    let hw = w * h;
    for (var c = 0; c < 3; c++) {
      var v: array<f32, 4>;
      for (var j = 0; j < 4; j++) {
        let raw = f32(chol[c * hw + b.idx[j]]);
        v[j] = select(softplus(raw), raw, c == 1);
      }
      res[at + 3 * i + c] = b.wt[0] * v[0] + b.wt[1] * v[1] + b.wt[2] * v[2] + b.wt[3] * v[3];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// XFeat (ported from kernels.metal; see tools/xfeat_folded.py)
// ---------------------------------------------------------------------------------------------

//@helpers xfeat
// XFeat's sampling positions: pixel p of a `full`-wide image, normalized as 2 * (p / (full - 1)) - 1 and read
// with align_corners = false from a map `size` wide. The division is rounded correctly: at p = full - 1 it
// must give exactly 1, which puts the nearest heatmap pixel outside the map, as on the other backends.
fn xfeat_source(p: i32, full: i32, size: i32) -> f32 {
  let g = 2.0 * div_rn(f32(p), f32(full - 1)) - 1.0;
  return ((g + 1.0) * f32(size) - 1.0) * 0.5;
}

//@kernel k_resize
// Bilinear, align_corners = false: source = max((dst + 0.5) * scale - 0.5, 0); sx, sy = in / out.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let c = pi(0u);
  let h = pi(1u);
  let w = pi(2u);
  let oh = pi(3u);
  let ow = pi(4u);
  let sx = pf(5u);
  let sy = pf(6u);
  let ohw = oh * ow;
  if (i >= c * ohw) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let xo = i % ow;
  let src_x = max(sx * (f32(xo) + 0.5) - 0.5, 0.0);
  let src_y = max(sy * (f32(y) + 0.5) - 0.5, 0.0);
  let x0 = min(i32(src_x), w - 1);
  let y0 = min(i32(src_y), h - 1);
  let x1 = x0 + select(0, 1, x0 < w - 1);
  let y1 = y0 + select(0, 1, y0 < h - 1);
  let w1 = clampf(src_x - f32(x0), 0.0, 1.0);
  let w0 = 1.0 - w1;
  let h1 = clampf(src_y - f32(y0), 0.0, 1.0);
  let h0 = 1.0 - h1;
  let r0 = (ch * h + y0) * w;
  let r1 = (ch * h + y1) * w;
  out[i] = real(h0 * (w0 * f32(x[r0 + x0]) + w1 * f32(x[r0 + x1])) + h1 * (w0 * f32(x[r1 + x0]) + w1 * f32(x[r1 + x1])));
}

//@kernel k_gray
// Mean of the channels (sum, then times 1 / c).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
fn run(i: i32) {
  let c = pi(0u);
  let hw = pi(1u);
  if (i >= hw) { return; }
  var sum = 0.0;
  for (var ch = 0; ch < c; ch++) { sum += f32(x[ch * hw + i]); }
  g[i] = sum * (1.0 / f32(c));
}

//@kernel k_moments_block
// Kahan sums in blocks of kScanBlock: mode 0 sums v, mode 1 sums (v - mean)^2 with mean = stats[0] / n.
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
fn run(b: i32) {
  let n = pi(0u);
  let mode = pi(1u);
  let begin = b * kScanBlock;
  if (begin >= n) { return; }
  let end = min(n, begin + kScanBlock);
  var mean = 0.0;
  if (mode != 0) { mean = div_rn(stats[0], f32(n)); }
  var sum = 0.0;
  var comp = 0.0;
  for (var i = begin; i < end; i++) {
    let d = v[i] - mean;
    let y = select(v[i], d * d, mode != 0) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  partial[b] = sum;
}

//@kernel k_moments_total
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
fn run(i: i32) {
  let blocks = pi(0u);
  let slot = pi(1u);
  if (i != 0) { return; }
  var sum = 0.0;
  var comp = 0.0;
  for (var b = 0; b < blocks; b++) {
    let y = partial[b] - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  stats[slot] = sum;
}

//@kernel k_instance_norm
// g * invstd - mean * invstd, invstd = 1 / sqrt(var + 1e-5) (biased variance), as PyTorch's batch norm.
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let n = pi(0u);
  if (i >= n) { return; }
  let mean = div_rn(stats[0], f32(n));
  let invstd = div_rn(1.0, sqrt(div_rn(stats[1], f32(n)) + 1e-5));
  out[i] = real(g[i] * invstd + (-mean * invstd));
}

//@kernel k_conv2d_s2_tile
// Stride-2 convolution, T output channels per thread for one output pixel, zero padding K / 2; weights as
// k_conv2d_tile. Grid: ceil(cout / T) * oh * ow.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let has_bias = pi(0u);
  let cin = pi(1u);
  let cout = pi(2u);
  let h = pi(3u);
  let w = pi(4u);
  let oh = pi(5u);
  let ow = pi(6u);
  let act = pi(7u);
  let ohw = oh * ow;
  let blocks = (cout + $T - 1) / $T;
  if (i >= blocks * ohw) { return; }
  let ob = i / ohw;
  let pix = i % ohw;
  let y = pix / ow;
  let xo = pix % ow;
  let pad = $K / 2;
  let oc0 = ob * $T;
  var acc: array<vec4<f32>, $T4>;
  for (var t = 0; t < $T; t++) {
    var b = 0.0;
    if (has_bias != 0 && oc0 + t < cout) { b = f32(bias[oc0 + t]); }
    acc[t / 4][t % 4] = b;
  }
  for (var ic = 0; ic < cin; ic++) {
    let base = ic * h * w;
    let wic = ic * $K * $K;
    for (var ky = 0; ky < $K; ky++) {
      let sy = 2 * y + ky - pad;
      if (sy < 0 || sy >= h) { continue; }
      for (var kx = 0; kx < $K; kx++) {
        let sx = 2 * xo + kx - pad;
        if (sx < 0 || sx >= w) { continue; }
        let v = vec4<f32>(f32(x[base + sy * w + sx]));
        let wp = ((wic + ky * $K + kx) * blocks + ob) * $T4;
        for (var q = 0; q < $T4; q++) { acc[q] = mac(acc[q], vec4<f32>(weight[wp + q]), v); }
      }
    }
  }
  for (var t = 0; t < $T; t++) {
    if (oc0 + t < cout) { out[(oc0 + t) * ohw + pix] = real(activate(acc[t / 4][t % 4], act)); }
  }
}

//@kernel k_space_to_depth
// 1xHxW -> (b*b)x(H/b)x(W/b); channel ky * b + kx holds pixel (b y + ky, b x + kx).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;
fn run(i: i32) {
  let h = pi(0u);
  let w = pi(1u);
  let b = pi(2u);
  let oh = h / b;
  let ow = w / b;
  let ohw = oh * ow;
  if (i >= h * w) { return; }
  let ch = i / ohw;
  let y = (i % ohw) / ow;
  let xo = i % ow;
  out[i] = x[(y * b + ch / b) * w + xo * b + ch % b];
}

//@kernel k_xfeat_heat
// Softmax over the 65 logits of each cell; the first 64 go to their pixels of the full-resolution heatmap.
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> heat: array<f32>;
fn run(i: i32) {
  let h8 = pi(0u);
  let w8 = pi(1u);
  let plane = h8 * w8;
  if (i >= plane) { return; }
  let y = i / w8;
  let x = i % w8;
  let W = 8 * w8;
  var m = kLowest;
  for (var c = 0; c < 65; c++) { m = max(m, logits[c * plane + i]); }
  var e: array<f32, 65>;
  var sum = 0.0;
  for (var c = 0; c < 65; c++) {
    e[c] = exp(logits[c * plane + i] - m);
    sum += e[c];
  }
  for (var c = 0; c < 64; c++) { heat[(8 * y + c / 8) * W + 8 * x + c % 8] = e[c] / sum; }
}

//@kernel k_xfeat_score
// For each local maximum above the threshold, the nearest heatmap value (round half to even, 0 outside)
// times the bilinear reliability; 0 elsewhere and at pixel (0, 0).
//@use xfeat
@group(0) @binding(1) var<storage, read> heat: array<f32>;
@group(0) @binding(2) var<storage, read> pooled: array<f32>;
@group(0) @binding(3) var<storage, read> rel: array<f32>;
@group(0) @binding(4) var<storage, read_write> score: array<f32>;
fn run(i: i32) {
  let H = pi(0u);
  let W = pi(1u);
  let th = pf(2u);
  if (i >= H * W) { return; }
  let px = i % W;
  let py = i / W;
  let h8 = H / 8;
  let w8 = W / 8;
  var s = 0.0;
  if (heat[i] == pooled[i] && heat[i] > th && (px != 0 || py != 0)) {
    let nx = i32(round(xfeat_source(px, W, W)));
    let ny = i32(round(xfeat_source(py, H, H)));
    var hv = 0.0;
    if (nx >= 0 && nx < W && ny >= 0 && ny < H) { hv = heat[ny * W + nx]; }
    let b = bilinear_setup(xfeat_source(px, W, w8), xfeat_source(py, H, h8), w8, h8);
    s = hv * (b.wt[0] * rel[b.idx[0]] + b.wt[1] * rel[b.idx[1]] + b.wt[2] * rel[b.idx[2]] + b.wt[3] * rel[b.idx[3]]);
  }
  score[i] = s;
}

//@kernel k_xfeat_describe
// Descriptors (before row normalization), scores and pixels of the selected keypoints. One thread per
// (keypoint, channel): bicubic sample (A = -0.75) of the normalized features, zeros outside.
//@use xfeat
@group(0) @binding(1) var<storage, read> f: array<real>;
@group(0) @binding(2) var<storage, read> score: array<f32>;
@group(0) @binding(3) var<storage, read> idx: array<i32>;
@group(0) @binding(4) var<storage, read_write> desc: array<f32>;
@group(0) @binding(5) var<storage, read_write> score_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> pix: array<i32>;
fn cubic_coeffs(t: f32) -> array<f32, 4> {
  let a = -0.75;
  let x0 = t + 1.0;
  let x2 = 1.0 - t;
  let x3 = 2.0 - t;
  return array<f32, 4>(((a * x0 - 5.0 * a) * x0 + 8.0 * a) * x0 - 4.0 * a, ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0,
                       ((a + 2.0) * x2 - (a + 3.0)) * x2 * x2 + 1.0, ((a * x3 - 5.0 * a) * x3 + 8.0 * a) * x3 - 4.0 * a);
}
fn run(t: i32) {
  let dim = pi(0u);
  let h8 = pi(1u);
  let w8 = pi(2u);
  let n = pi(3u);
  if (t >= n * dim) { return; }
  let i = t / dim;
  let ch = t % dim;
  let H = 8 * h8;
  let W = 8 * w8;
  let px = idx[i] % W;
  let py = idx[i] / W;
  if (ch == 0) {
    score_out[i] = score[idx[i]];
    pix[2 * i] = px;
    pix[2 * i + 1] = py;
  }
  let sx = xfeat_source(px, W, w8);
  let sy = xfeat_source(py, H, h8);
  let fx = floor(sx);
  let fy = floor(sy);
  let x0 = i32(fx) - 1;
  let y0 = i32(fy) - 1;
  let cx = cubic_coeffs(sx - fx);
  let cy = cubic_coeffs(sy - fy);
  let base = ch * h8 * w8;
  var v = 0.0;
  for (var a = 0; a < 4; a++) {
    let yy = y0 + a;
    var tap: array<f32, 4>;
    for (var b = 0; b < 4; b++) {
      let xx = x0 + b;
      tap[b] = 0.0;
      if (xx >= 0 && xx < w8 && yy >= 0 && yy < h8) { tap[b] = f32(f[base + yy * w8 + xx]); }
    }
    let row = tap[0] * cx[0] + tap[1] * cx[1] + tap[2] * cx[2] + tap[3] * cx[3];
    v += row * cy[a];
  }
  desc[i * dim + ch] = v;
}
