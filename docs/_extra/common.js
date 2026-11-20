// Shared by the live demo and the benchmark page: what the browser can run, the weights, image preparation.
import * as jkp from './lib/jkp.js';

export { jkp };

export const WEIGHTS_URL = new URL('./lib/aliked-n16.jkpw', import.meta.url).href;
export const RACO_WEIGHTS_URL = new URL('./lib/raco.jkpw', import.meta.url).href;
export const XFEAT_WEIGHTS_URL = new URL('./lib/xfeat.jkpw', import.meta.url).href;

// What this browser can run. The module waits for the GPU through JSPI (WebAssembly.Suspending); without
// it the module cannot run at all. Without WebGPU the module's CPU backend still runs, slowly.
export async function environment() {
  const env = { jspi: typeof WebAssembly === 'object' && 'Suspending' in WebAssembly, webgpu: false, f16: false,
                adapter: '' };
  if (jkp.webgpuSupported()) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        env.webgpu = true;
        env.f16 = adapter.features.has('shader-f16');
        const info = adapter.info ?? {};
        env.adapter = [info.vendor, info.architecture, info.device, info.description]
          .filter((s, i, all) => s && all.indexOf(s) === i).join(' · ') || 'unnamed adapter';
      }
    } catch {
      env.webgpu = false;
    }
  }
  return env;
}

const weightsPromises = new Map();

// A weights file, fetched once per page: ALIKED's by default, RaCo's with RACO_WEIGHTS_URL.
export function weights(url = WEIGHTS_URL) {
  if (!weightsPromises.has(url)) {
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`cannot load the weights: HTTP ${r.status}`);
      return r.arrayBuffer();
    }).then((b) => new Uint8Array(b));
    p.catch(() => weightsPromises.delete(url));
    weightsPromises.set(url, p);
  }
  return weightsPromises.get(url);
}

export function sourceSize(source) {
  return { w: source.naturalWidth ?? source.videoWidth ?? source.width,
           h: source.naturalHeight ?? source.videoHeight ?? source.height };
}

// RGBA pixels of `source` at w x h. crop: centre-crop to the target's aspect ratio first (the benchmark),
// otherwise the whole image is scaled.
export function pixels(source, w, h, crop = false) {
  const s = sourceSize(source);
  let sx = 0, sy = 0, sw = s.w, sh = s.h;
  if (crop) {
    if (s.w * h > s.h * w) { sw = Math.round(s.h * w / h); sx = Math.floor((s.w - sw) / 2); }
    else { sh = Math.round(s.w * h / w); sy = Math.floor((s.h - sh) / 2); }
  }
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingQuality = 'high';
  g.drawImage(source, sx, sy, sw, sh, 0, 0, w, h);
  return g.getImageData(0, 0, w, h);
}

// fetch and createImageBitmap rather than an <img>: image decoding of an <img> can wait while the page is in
// the background.
export async function loadImage(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`cannot load ${url}: HTTP ${r.status}`);
  return createImageBitmap(await r.blob(), { imageOrientation: 'from-image' });
}

export function median(values) {
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function ms(value) {
  if (value == null || !isFinite(value)) return '–';
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

// The message shown when the module cannot run here, or null.
export function unsupportedMessage(env) {
  if (!env.jspi) {
    return 'This browser cannot run the module: it needs WebAssembly JavaScript promise integration (JSPI), ' +
           'which Chrome and Edge have from version 137.';
  }
  return null;
}
