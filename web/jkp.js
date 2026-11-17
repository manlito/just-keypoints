// just-keypoints in the browser: a thin wrapper over the WebAssembly module (jkp_wasm.js + .wasm).
//
//   import * as jkp from './jkp.js';
//   const extractor = await jkp.create({ weightsUrl: 'aliked-n16.jkpw' });
//   const { n, dim, xy, score, desc } = await extractor.extract(imageBitmap);
//
// Results are copies (Float32Array): xy [n][2] in pixels, (0, 0) = centre of the top-left pixel; score [n];
// desc [n][dim], L2-normalized. RaCo (weights raco.jkpw) adds rank [n], the ranker score (higher is better),
// and cov [n][2][2], the keypoint covariance in pixels^2; both are null for ALIKED or a skipped head. RaCo
// has no descriptors of its own: give descriptorWeights(Url) of aliked-n16 to describe its keypoints.
//
//   const raco = await jkp.create({ weightsUrl: 'raco.jkpw', descriptorWeightsUrl: 'aliked-n16.jkpw' });
//
// The module waits for the GPU through JSPI (Chrome, Edge).
import createJkp from './jkp_wasm.js';

const BACKENDS = { cpu: 0, cuda: 1, metal: 2, webgpu: 3 };
const PRECISIONS = { fp32: 0, fp16: 1 };

let modulePromise = null;

// The module, loaded once per page.
export function module() {
  if (!modulePromise) modulePromise = createJkp();
  return modulePromise;
}

export function webgpuSupported() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

// RGBA pixels of an ImageBitmap, ImageData, image element, video frame or canvas.
export function rgba(image) {
  if (typeof ImageData !== 'undefined' && image instanceof ImageData) return image;
  const w = image.width ?? image.videoWidth, h = image.height ?? image.videoHeight;
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.drawImage(image, 0, 0);
  return g.getImageData(0, 0, w, h);
}

// options: weightsUrl or weights (ArrayBuffer / Uint8Array of a .jkpw file); backend 'webgpu' (default)
// or 'cpu'; precision 'fp32' (default) or 'fp16' (needs shader-f16; ALIKED only); topK (> 0: keep the best
// K) or scoreThreshold (0.2) and maxKeypoints (5000). RaCo: always the best topK (2048 by default);
// descriptorWeightsUrl or descriptorWeights (aliked-n16) for descriptors; skipRanker, skipCovariance.
export async function create(options = {}) {
  const M = await module();
  const backend = options.backend ?? 'webgpu';
  const precision = options.precision ?? 'fp32';
  if (!(backend in BACKENDS)) throw new Error(`unknown backend '${backend}'`);
  if (!(precision in PRECISIONS)) throw new Error(`unknown precision '${precision}'`);
  if (!M._jkpw_backend_available(BACKENDS[backend])) throw new Error(`backend '${backend}' is not in this build`);
  if (!options.weights && !options.weightsUrl) throw new Error('give weightsUrl or weights');
  const bytes = await load(options.weights, options.weightsUrl);
  const dbytes = await load(options.descriptorWeights, options.descriptorWeightsUrl);
  const wp = M._malloc(bytes.length), dp = dbytes ? M._malloc(dbytes.length) : 0;
  M.HEAPU8.set(bytes, wp);
  if (dbytes) M.HEAPU8.set(dbytes, dp);
  let handle;
  try {
    handle = await M._jkpw_create(BACKENDS[backend], PRECISIONS[precision], wp, bytes.length, dp, dbytes ? dbytes.length : 0,
                                  options.topK ?? -1, options.scoreThreshold ?? 0.2, options.maxKeypoints ?? 5000,
                                  options.skipRanker ? 1 : 0, options.skipCovariance ? 1 : 0);
  } finally {
    M._free(wp);  // jkp_create copies the weights
    if (dp) M._free(dp);
  }
  if (!handle) throw new Error(M.UTF8ToString(M._jkpw_error()));
  return new Extractor(M, handle, backend, precision);
}

// A .jkpw file as bytes: given (ArrayBuffer / Uint8Array), fetched from url, or null when neither is set.
async function load(data, url) {
  if (!data && !url) return null;
  if (!data) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`cannot load ${url}: ${response.status}`);
    data = await response.arrayBuffer();
  }
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

class Extractor {
  constructor(M, handle, backend, precision) {
    this.M = M;
    this.handle = handle;
    this.backend = backend;
    this.precision = precision;
    this.pixels = 0;
    this.capacity = 0;
  }

  // One image at a time per extractor: calls must not overlap.
  async extract(image) {
    const M = this.M;
    if (!this.handle) throw new Error('extractor destroyed');
    const data = rgba(image);
    const bytes = data.width * data.height * 4;
    if (bytes > this.capacity) {
      if (this.pixels) M._free(this.pixels);
      this.pixels = M._malloc(bytes);
      this.capacity = bytes;
    }
    M.HEAPU8.set(data.data, this.pixels);
    const n = await M._jkpw_extract(this.handle, this.pixels, data.width, data.height);
    if (n < 0) throw new Error(M.UTF8ToString(M._jkpw_error()));
    const dim = M._jkpw_dim(this.handle);
    const view = (ptr, count) => new Float32Array(M.HEAPF32.buffer, ptr, count).slice();
    const rank = M._jkpw_rank(this.handle), cov = M._jkpw_cov(this.handle);
    return {
      n,
      dim,
      xy: view(M._jkpw_xy(this.handle), 2 * n),
      score: view(M._jkpw_score(this.handle), n),
      desc: dim > 0 ? view(M._jkpw_desc(this.handle), n * dim) : new Float32Array(0),
      rank: rank ? view(rank, n) : null,
      cov: cov ? view(cov, 4 * n) : null,
    };
  }

  destroy() {
    if (!this.handle) return;
    this.M._jkpw_destroy(this.handle);
    if (this.pixels) this.M._free(this.pixels);
    this.handle = 0;
    this.pixels = 0;
  }
}
