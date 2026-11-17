# In the browser

The browser build is the same C++ library compiled to WebAssembly, with the WebGPU backend running the network
on the visitor's GPU. A small JavaScript wrapper, `jkp.js`, loads it.

```js
import * as jkp from './jkp.js';

const extractor = await jkp.create({ weightsUrl: 'aliked-n16.jkpw', topK: 1000 });
const bitmap = await createImageBitmap(file);
const { n, dim, xy, score, desc } = await extractor.extract(bitmap);
// xy: Float32Array [n][2], score: [n], desc: [n][dim], L2-normalized
extractor.destroy();
```

Serve `jkp.js`, `jkp_wasm.js`, `jkp_wasm.wasm` and the weights file (2.7 MB) as static files; nothing runs on a
server. <a href="demo/index.html">The live demo</a> is such a page.

RaCo keypoints come from `raco.jkpw` (2 MB). RaCo has no descriptors of its own; give it ALIKED's weights as well
to describe its keypoints for matching:

```js
const raco = await jkp.create({ weightsUrl: 'raco.jkpw', descriptorWeightsUrl: 'aliked-n16.jkpw', topK: 2048 });
const { n, xy, score, rank, cov, desc } = await raco.extract(bitmap);
// rank: [n] ranker score, higher is better; cov: [n][2][2] covariance in pixels²
```

To keep fewer keypoints, sort by `rank` and keep the best: RaCo's authors subsample by ranker score, not by
detection score.

## Options of `create`

| Option | Default | Meaning |
|---|---|---|
| `weightsUrl` or `weights` | | URL of the weights file, or its bytes (`ArrayBuffer`, `Uint8Array`) |
| `backend` | `'webgpu'` | `'webgpu'`, or `'cpu'` (the CPU backend in WebAssembly: one thread, about 1.2 s at 640x480) |
| `precision` | `'fp32'` | `'fp16'` needs the adapter's `shader-f16` feature; ALIKED only |
| `topK` | `-1` | `> 0`: keep the K best keypoints. RaCo: always the best K, 2048 when not set |
| `scoreThreshold`, `maxKeypoints` | `0.2`, `5000` | threshold mode, when `topK` is not set (ALIKED) |
| `descriptorWeightsUrl` or `descriptorWeights` | | RaCo: ALIKED's weights, to describe its keypoints |
| `skipRanker`, `skipCovariance` | `false` | RaCo: leave out a head; `rank` or `cov` is then `null` |

Results are copies: `xy`, `score`, `desc` (empty without descriptors), and for RaCo `rank` and `cov`
(`null` for ALIKED).

`extract` takes an `ImageBitmap`, `ImageData`, image element, video frame or canvas. Calls on one extractor must
not overlap; create a second extractor to run two images at once.

## Browser support

The module waits for the GPU through JavaScript promise integration (JSPI), so that the C++ code can wait for
results without blocking the page. Chrome and Edge 137 and later have both WebGPU and JSPI. Where WebGPU is
missing but JSPI is present, the CPU backend still runs, slowly.

The FP32 results in Chrome are the native CPU backend's keypoints on every image tried (positions within
1.4e-4 px); FP16 finds 99.6-100 % of them within 1 px.

## Sharing a GPU device

An app that also does its own WebGPU work can give the library its device (`config.webgpu_instance` and
`config.webgpu_device` in the C API) instead of letting it create one. An app that builds its own WebAssembly
module links the static library and calls the C API directly, with the same settings: JSPI and
`-fwasm-exceptions`.
