# just-keypoints — plan

Minimal-dependency ALIKED keypoint + descriptor extraction for bundle-adjustment pipelines.

## Decisions

| Topic | Choice |
|---|---|
| Model | ALIKED `aliked-n16` (128-dim descriptors, BSD-3 weights) |
| Scope | Keypoints + descriptors only. No matcher. |
| Precision | FP32 "reference" mode on every backend, matching the original ALIKED repo; it is the default. FP16 is an opt-in fast mode (`config.precision`). |
| Weights embedding | User's choice at build time (`JKP_EMBED_WEIGHTS=ON/OFF`, per model). Loading from a path, memory blob or URL always works. |
| Extensibility | Other detectors can be added later: model id in config and weights file, per-model graph under `src/models/`, shared op kernels. Only ALIKED is implemented now. |
| Native backend | Hand-written CUDA kernels. No PyTorch, ONNX Runtime, cuDNN or TensorRT. |
| Apple backend | Hand-written Metal compute shaders (MSL). No Core ML or MPSGraph. |
| Web backend | WGSL compute shaders, C++ core compiled to WASM (Emscripten) |
| Structure | One C++ core (weights, graph, API) + pluggable backends: `cpu`, `cuda`, `metal`, `webgpu` |

## Why hand-written kernels are feasible

ALIKED is a small, fixed graph. The whole op set is:

- 3x3 and 1x1 convolution (BatchNorm folded into weights at export time)
- SELU / ReLU activation
- Average pooling (2x, 4x)
- Bilinear upsampling
- Deformable 3x3 convolution (later encoder blocks) — bilinear sampling at learned offsets
- DKD: score-map NMS, top-K selection, soft-argmax subpixel refinement
- SDDH: per-keypoint descriptor head — sample features at learned offsets around each keypoint, aggregate, L2-normalize

Every one is a data-parallel kernel that maps 1:1 from a CUDA kernel to a Metal or WGSL compute shader.
SDDH runs only on the K detected keypoints, not on the dense map.

## Repository layout

```
include/jkp/jkp.h        C API (the only public header)
src/core/                weights loader, tensor/buffer abstractions, model registry
src/models/aliked/       ALIKED graph (built from shared ops); future detectors get a sibling folder
src/backend/cpu/         plain C++ reference (correctness oracle, WASM fallback)
src/backend/cuda/        .cu kernels
src/backend/metal/        .metal shaders + Objective-C++ glue
src/backend/webgpu/      .wgsl shaders + thin C++ glue (Dawn native / Emscripten)
tools/export_weights.py  dev-only: PyTorch checkpoint -> flat .jkpw file
tools/dump_golden.py     dev-only: per-layer activations for test images
tools/compare.py         dev-only: run original ALIKED and jkp on the same images, report differences
tests/                   golden comparison tests per backend
web/                     JS wrapper (npm package)
docs/                    documentation site + live demo (static, GitHub Pages)
```

Python/PyTorch is used only in `tools/` to produce weights and test vectors. It is never a runtime dependency.

## Public API (C, plain buffers)

```c
jkp_ctx*  jkp_create(const jkp_config*);          // backend, weights path/blob, max_keypoints, score_threshold
int       jkp_extract(jkp_ctx*, const jkp_image*, jkp_result*);
void      jkp_destroy(jkp_ctx*);
```

- **Input** `jkp_image`: pointer + width/height/stride/format (`GRAY8`, `RGB8`, `RGBA8`).
  Host pointer by default; optional CUDA device pointer or `MTLTexture` (zero-copy) and, on web, a `GPUTexture` / `ImageBitmap`.
  Normalization and padding to a multiple of 32 happen on the GPU.
- **Output** `jkp_result` (struct of arrays, caller- or library-owned):
  - `xy`     float32 `[N,2]` — pixel coordinates in the original image, subpixel
  - `score`  float32 `[N]`
  - `desc`   float32 `[N,D]`, L2-normalized; `D` reported in the result (128 for `aliked-n16`) (optional fp16 / uint8-quantized output later)
  - `n`      number of keypoints (<= `max_keypoints`)
- Results can stay on the device (CUDA pointer / `MTLBuffer` / `GPUBuffer`) for a downstream GPU matcher.

## Weights format (`.jkpw`)

Single flat binary: magic + version + model id + tensor table (name, shape, dtype, offset) + raw little-endian data.
The model id selects the graph, so one loader serves every future detector.
No protobuf, no parsing dependencies. Loadable via `fread`, `mmap`, or `fetch()` → `ArrayBuffer`.
One FP32 file is shipped; FP16 mode converts the weights at load time.

## Milestones

Each milestone is a state you can run and check. Phases below hold the technical detail.

- [ ] **M1 — Reference locked** (phase 0)
  Original ALIKED runs under PyTorch on CPU; weights exported to `.jkpw`; golden vectors and `compare.py` exist.
- [ ] **M2 — First keypoints from our own code** (phase 1)
  `jkp extract image.png` (CPU, C++, no dependencies) produces keypoints and descriptors that match the original in `compare.py`.
- [ ] **M3 — Runs on the Mac GPU** (phase 4, FP32)
  Metal backend passes the same comparison on the M5 laptop.
- [ ] **M4 — Runs on NVIDIA** (phase 2)
  CUDA backend passes the same comparison. Order of M3/M4 depends on the machine in use.
- [ ] **M5 — Fast** (phase 3, plus FP16 on Metal)
  Benchmark table: jkp CUDA vs PyTorch CUDA, jkp Metal, FP32 vs FP16, at three resolutions. FP16 accuracy loss measured.
- [ ] **M6 — Runs in the browser** (phase 5)
  WebGPU backend passes the comparison natively (Dawn) and in a browser test page.
- [ ] **M7 — Public demo and docs** (phase 6)
  GitHub Pages site: load two images, see keypoints, matches and timings.
- [ ] **M8 — v1.0** (phase 7)
  CMake package, npm package, tagged release with weights and benchmark numbers.

## Phases

### 0 — Reference and golden data
- Pin the upstream ALIKED reference commit and the `aliked-n16` checkpoint; record license.
- `export_weights.py`: fold BatchNorm, write `.jkpw`.
- `dump_golden.py`: for a few test images, dump every intermediate tensor plus final keypoints/descriptors.
- `compare.py`: runs the original ALIKED repo (PyTorch, CPU or CUDA) and any jkp backend on arbitrary images and reports
  keypoint position error, score error, descriptor cosine similarity, and keypoint-set overlap.
  PyTorch on CPU is enough for validation; PyTorch on CUDA is also the speed baseline for phase 3.
- **Exit:** weights file + golden vectors checked in (or fetched by script).

### 1 — CPU reference in C++
- Core: weights loader, graph description, C API.
- Naive, readable implementation of every op.
- Small CLI (`jkp extract`, `jkp bench`) that works with every backend.
- **Exit:** per-layer match with golden (max abs error ~1e-4); final keypoints within 0.01 px, descriptor cosine > 0.999.

### 2 — CUDA backend (correctness)
- One kernel per op, FP32, NCHW, same graph as CPU.
- GPU top-K and compaction so only N keypoints are copied back.
- **Exit:** passes the same golden tests as CPU.

### 3 — CUDA backend (speed)
- Benchmark harness (per-op timing, 640x480 / 1024x768 / 1920x1080).
- Opt-in FP16 storage and math (FP32 reference mode stays available and tested), fused conv+activation, preallocated arena (zero allocations per frame),
  specialized 3x3 kernels per channel count, pinned-memory upload, optional batch of images.
- Compare against the PyTorch reference and existing ALIKED CUDA/TensorRT ports as baselines.
- **Exit:** documented ms/frame numbers; accuracy regression test (FP16 vs FP32 golden).

### 4 — Metal backend
- Port each kernel to a Metal compute shader; same graph, same golden tests.
- Shared memory on Apple Silicon: `MTLBuffer` with shared storage, no upload/readback copies.
- FP32 reference mode first, then opt-in FP16 (`half`); reuse the phase 3 benchmark harness.
- **Exit:** passes golden tests on the M5 laptop; documented ms/frame numbers.

### 5 — WebGPU backend
- Port each kernel to WGSL; same graph, same tests.
- Run natively through Dawn first (fast iteration, same golden tests in CI), then build with Emscripten.
- JS wrapper: `await jkp.create({weightsUrl})`, `await jkp.extract(imageBitmap)` → typed arrays.
- FP32 reference mode; opt-in FP16 when `shader-f16` is available.
- **Exit:** minimal test page; results match golden within FP16 tolerance.

### 6 — Documentation site and live demo
- Static site in `docs/`, built and deployed to GitHub Pages by CI. No server: wasm, shaders and weights are static assets.
- Content: quick start per backend (C, CUDA, Metal, web), API reference, weights format, benchmark tables.
- **Live demo** (runs fully in the visitor's browser on WebGPU):
  - Pick two images (samples, upload, or webcam); keypoints drawn on both.
  - Match lines between the pair. The matcher is a small mutual-nearest-neighbor + ratio test living in the demo code only — the library scope stays keypoints + descriptors.
  - Controls: max keypoints, score threshold, resolution, FP16/FP32.
  - Performance panel: ms per stage (upload, network, top-K, descriptors), keypoints/frame, GPU adapter name.
- **Local benchmark page:** runs a fixed image set N times and reports ms/frame, so numbers are comparable across machines.
- Fallback message when WebGPU is unavailable.
- **Exit:** public page where two images can be loaded and matched, with timings shown.

### 7 — Packaging
- CMake install target + `find_package` config; single header.
- npm package (wasm + js + weights).
- Optional thin Python binding (ctypes over the C API) for use in SfM scripts.

## Development machines

- Phases 0, 1, 4, 5, 6, 7 run on the M5 laptop (WebGPU via Dawn uses Metal there).
- Phases 2 and 3 need an NVIDIA GPU machine.
- The CUDA and Metal phases only depend on phase 1, so either can go first depending on the machine in use.

## Open questions (decide during implementation)

- Tensor layout for speed: NCHW vs NHWC on CUDA; texture vs storage-buffer representation in WGSL.
- WebGPU limits (storage buffer size, bindings per shader stage) at large resolutions — may need tiling.
- Whether to add `aliked-t16` as a second, faster web-friendly variant (same op set, different widths).
- Descriptor output quantization for the downstream matcher.
