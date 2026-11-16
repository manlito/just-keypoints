# just-keypoints — plan

Minimal-dependency ALIKED keypoint + descriptor extraction for bundle-adjustment pipelines.

## Decisions

| Topic | Choice |
|---|---|
| Model | ALIKED `aliked-n16` (128-dim descriptors, BSD-3 weights) |
| Scope | Keypoints + descriptors only. No matcher. |
| Precision | FP32 "reference" mode on every backend, matching the original ALIKED repo; it is the default. FP16 is an opt-in fast mode (`config.precision`). |
| Weights embedding | User's choice at build time (`JKP_EMBED_WEIGHTS=ON/OFF`, per model). Loading from a path, memory blob or URL always works. |
| Second detector | RaCo (cvg/RaCo, Apache-2.0), after ALIKED is complete. See "RaCo" below. No further detectors planned. |
| Extensibility | Model id in config and weights file, per-model graph under `src/models/`, shared op kernels. Detector and descriptor are separate stages so they can be combined. |
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
  - `rank`   float32 `[N]` — optional, RaCo ranker score
  - `cov`    float32 `[N,2,2]` — optional, RaCo keypoint covariance in pixels (usable as a BA weight)
- `desc` is optional too (`D = 0` for a detector without descriptors).
- Config selects `detector` and `descriptor` independently, e.g. `detector = raco`, `descriptor = aliked-n16`.
- Results can stay on the device (CUDA pointer / `MTLBuffer` / `GPUBuffer`) for a downstream GPU matcher.

## Weights format (`.jkpw`)

Single flat binary: magic + version + metadata (model id, widths, K, M) + tensor table (name, shape, dtype, offset) + raw little-endian data.
Exact layout: see the top of `tools/jkpfile.py`.
The model id selects the graph, so one loader serves every future detector.
No protobuf, no parsing dependencies. Loadable via `fread`, `mmap`, or `fetch()` → `ArrayBuffer`.
One FP32 file is shipped; FP16 mode converts the weights at load time.

## Milestones

Each milestone is a state you can run and check. Phases below hold the technical detail.

- [x] **M1 — Reference locked** (phase 0)
  Original ALIKED runs under PyTorch on CPU; weights exported to `.jkpw`; golden vectors and `compare.py` exist.
  Done: upstream pinned at `683d7c6`; `weights/aliked-n16.jkpw` is 44 tensors, 676,876 values, 2.71 MB.
  The reference forward pass from that file (`tools/folded.py`) matches the original on 27 images:
  identical keypoint sets, position error <= 1.3e-4 px, score error <= 3e-5, descriptor cosine >= 1 - 2e-10.
- [x] **M2 — First keypoints from our own code** (phase 1)
  `jkp extract image.png` (CPU, C++, no dependencies) produces keypoints and descriptors that match the original in `compare.py`.
  Done: 31/31 sample images match the original (identical keypoint sets, position error <= 1.3e-4 px,
  score error <= 5e-5, descriptor cosine >= 1 - 5e-10); every intermediate tensor matches the golden dump
  within 2e-5. Speed is not a goal here: ~0.8 s for 640x480 on 4 CPU threads.
- [x] **M3 — Runs on the Mac GPU** (phase 4, FP32)
  Metal backend passes the same comparison on the M5 laptop.
  Done (M5 Pro, macOS 26): `jkp extract --backend metal` matches the original on 57/57 sample images,
  in threshold and top-k mode (identical keypoint sets, position error <= 2.6e-4 px, score error <= 8e-5,
  descriptor cosine >= 1 - 4e-9); every intermediate tensor matches the golden dump within 3.1e-5 and the
  CPU backend within 2.3e-5. Not tuned yet: 25 ms for 640x400, 60-75 ms for 1024x640, 230 ms for
  1920x1200, about 2x the CPU backend on the same machine.
- [x] **M4 — Runs on NVIDIA** (phase 2)
  CUDA backend passes the same comparison. Order of M3/M4 depends on the machine in use.
  Done (RTX 2070, CUDA 13.4): `jkp extract --backend cuda` matches the original on 57/57 sample images,
  in threshold and top-k mode (identical keypoint sets, position error <= 2.6e-4 px, score error <= 7e-5,
  descriptor cosine >= 1 - 2e-9); every intermediate tensor matches the golden dump within 3e-5 and the
  CPU backend within 2e-5. Not tuned yet: 43 ms for 640x480, about 8x the CPU backend on 12 threads.
- [ ] **M5 — Fast** (phase 3, plus FP16 on Metal)
  Benchmark table: jkp CUDA vs PyTorch CUDA, jkp Metal, FP32 vs FP16, at three resolutions. FP16 accuracy loss measured.
- [ ] **M6 — Runs in the browser** (phase 5)
  WebGPU backend passes the comparison natively (Dawn) and in a browser test page.
- [ ] **M7 — Public demo and docs** (phase 6)
  GitHub Pages site: load two images, see keypoints, matches and timings.
- [ ] **M8 — v1.0** (phase 7)
  CMake package, npm package, tagged release with weights and benchmark numbers.
- [ ] **M9 — RaCo** (phase 8)
  RaCo keypoints, ranker scores and covariances match the original on every backend; RaCo + ALIKED descriptors selectable in the demo.

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

### 8 — RaCo detector
- Export `raco.pth` to `.jkpw`, add `tools/` reference pass and `compare.py` support (same method as M1).
- `src/models/raco/` graph on CPU, then CUDA, Metal, WebGPU.
- "Describe at given keypoints" entry in the ALIKED descriptor head, so RaCo keypoints get ALIKED descriptors.
- Demo: detector switch, covariance ellipses, top-K by ranker score.
- **Exit:** M9.

## RaCo

Source: https://github.com/cvg/RaCo (3DV 2026, Apache-2.0), reviewed at commit `ee06fa6`.

- **Detector only.** Outputs keypoints, detection scores, ranker scores and 2x2 covariances. No descriptors:
  the authors match with ALIKED descriptors + LightGlue. For this repo that means running the ALIKED
  network as well to describe RaCo keypoints — roughly two networks per image.
- **Backbone** is ALIKED-n16 widths (16/32/64/128) with plain 3x3 convolutions, no deformable convolution.
  Same pooling, upsampling, SELU and score head as ALIKED, so most kernels are reused.
- **New ops:** reflect padding, 5x5 convolution, LeakyReLU, Softplus, global softmax over the score map,
  ImageNet mean/std input normalization.
- **Ranker head** runs on the full-resolution image: 9 residual blocks of 12 channels. Likely the most
  expensive part; it can be skipped when ranking is not needed.
- **Covariance head:** three 3x3 convolutions on the aggregated features, 3 Cholesky elements per pixel,
  sampled at keypoints.
- **Detection:** max-pool NMS (radius 3), top-K (default 2048), subpixel refinement; keypoints use a
  pixel-centre convention (+0.5) that differs from ALIKED and must be unified in our API.
- Weights are a release asset (`v1.0.0/raco.pth`); confirm the weights fall under the repo's Apache-2.0
  license before redistributing an exported file.

## Development machines

- Phases 0, 1, 4, 5, 6, 7 run on the M5 laptop (WebGPU via Dawn uses Metal there).
- Phases 2 and 3 need an NVIDIA GPU machine.
- The CUDA and Metal phases only depend on phase 1, so either can go first depending on the machine in use.

## Notes from M1 (for the C++ port)

- `tools/folded.py` is the executable specification: it uses only the `.jkpw` file and plain tensor ops.
- Keypoints are stored normalized to [-1, 1] and the descriptor head truncates the pixel position recovered
  from that value. The C++ code must keep the same float32 operation order, or a keypoint near an integer
  pixel can pick a different 3x3 patch.
- In threshold mode keypoints come out in raster order; only when more than `n_limit` pass the threshold
  are they sorted by score and cut.
- Golden files are ~50-100 MB per image at 320 px (dense feature maps), so they are generated locally
  with `tools/dump_golden.py`, not committed.
- The original's `get_patches` custom op needs CUDA to build; `tools/upstream.py` has an equivalent CPU version.

## Notes from M2

- The ALIKED graph (`src/models/aliked/`) is written once against `src/core/ops.h`; a GPU backend implements
  that interface (about a dozen operations, including `detect` and `describe`). Expect to revise the interface
  when the first GPU backend shows what it needs (preallocation, fused operations).
- The library is built with `-ffp-contract=off` so results do not depend on FMA availability.
- The CLI decodes images with a vendored `stb_image.h`; the library itself has no image-decoding code.
- `tests/golden_test.cpp` checks every stage against a golden dump and is the main debugging tool for new backends.

## Notes from M4

- `src/backend/cuda/cuda_ops.cu` holds every kernel and the host glue; the only dependency is the CUDA
  runtime library. The `Ops` interface did not need to change.
- Kernels keep the float32 operation order of the CPU backend and are compiled with `--fmad=false`, so the
  two backends differ only through `exp()` (SELU, sigmoid, soft-argmax). That is what makes the keypoint
  sets identical; keep it when optimizing (M5), or accept and measure the difference there.
- Selection runs on the device: block-wise ordered compaction for threshold mode; radix select on the
  float bits, then a rank sort of the kept keypoints, for top-k and for the cap. Ties resolve in raster
  order as on the CPU. Read back per image: N keypoints, N descriptors and a few counters.
- Known costs left for M5: one thread per output value with no fusing or tiling, weights read from global
  memory, the rank sort is O(N^2) (fine for N <= ~20k), keypoints make a host round trip between
  `detect` and `describe`, and the mean-score fallback sums the score map in a single GPU thread.
- `tests/backend_test.cpp` compares CUDA with CPU on generated images (selection modes, pixel formats,
  flat and tiny images) and needs no golden files; `golden_test` takes the backend as a third argument.
  A Metal or WebGPU backend can reuse both.

## Notes from M3

- `src/backend/metal/kernels.metal` is a kernel-by-kernel port of the CUDA file (same names, same argument
  order, one scalar per buffer index via `setBytes`); `metal_ops.mm` is the host glue. Dependencies:
  Metal.framework and Foundation only. The `Ops` interface did not change.
- The shader source is embedded in the library (`cmake/embed_text.cmake`) and compiled at `jkp_create`
  with `newLibraryWithSource`, so building needs no Metal toolchain (`xcodebuild -downloadComponent
  MetalToolchain` is not installed on the M5). Compile options: safe math mode, precise math functions,
  and `#pragma METAL fp contract(off)` in the source. As on CUDA, results differ from the CPU backend only
  through `exp()`. Precompiling a `.metallib` (when the toolchain is present) would save the compile at startup.
- All buffers use shared storage: the image is written and results are read in place, with no copies.
  Work is encoded into one command buffer with serial dispatch and committed only when the host needs a
  value (stage readback, selection counts, results). The buffer pool hands a buffer released during the
  open command buffer back to GPU work at once (dispatch order makes that safe) but never to a buffer the
  host writes; those wait for the next sync.
- Selection matches CUDA: block-wise ordered compaction, radix select on the float bits, rank sort. The
  prefix sum over block counts and the mean-score fallback sum run on the host (shared memory makes that
  free and the fallback sum identical to the CPU backend).
- `tests/backend_test.cpp` now takes `cuda|metal`; ctest registers `<backend>_vs_cpu` for each GPU backend
  in the build and the golden tests per backend.
- Known costs left for M5: same as CUDA (no tiling or fusing, O(N^2) rank sort, keypoint round trip between
  `detect` and `describe`), plus a full sync for each radix-select pass. Peak memory is ~1.7 GB at
  1024x640, mostly the deformable-convolution column buffer and the pool keeping every size it has seen.

## Open questions (decide during implementation)

- Tensor layout for speed: NCHW vs NHWC on CUDA; texture vs storage-buffer representation in WGSL.
- WebGPU limits (storage buffer size, bindings per shader stage) at large resolutions — may need tiling.
- Whether to add `aliked-t16` as a second, faster web-friendly variant (same op set, different widths).
- Descriptor output quantization for the downstream matcher.
