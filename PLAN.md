# just-keypoints — plan

Minimal-dependency keypoint extraction for bundle-adjustment pipelines: ALIKED keypoints + descriptors,
and the RaCo detector (keypoints with ranker scores and covariances).

## Decisions

| Topic | Choice |
|---|---|
| Model | ALIKED `aliked-n16` (128-dim descriptors, BSD-3 weights) |
| Scope | Keypoints + descriptors only. No matcher. |
| Precision | FP32 "reference" mode on every backend, matching the original ALIKED repo; it is the default. FP16 is an opt-in fast mode (`config.precision`). |
| Weights embedding | User's choice at build time (`JKP_EMBED_WEIGHTS=ON/OFF`, per model). Loading from a path, memory blob or URL always works. |
| Second detector | RaCo (cvg/RaCo, Apache-2.0), developed alongside the backends instead of after them: it runs on the CPU backend now and reaches each GPU backend once that backend passes ALIKED. See "RaCo track" and "RaCo" below. No further detectors planned. |
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
src/models/raco/         RaCo graph, same shared ops plus three of its own
tools/export_weights.py  dev-only: PyTorch checkpoint -> flat .jkpw file
tools/dump_golden.py     dev-only: per-layer activations for test images
tools/compare.py         dev-only: run the original (ALIKED or RaCo) and jkp on the same images, report differences
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
  - `rank`   float32 `[N]` — RaCo ranker score, higher is better; `NULL` for ALIKED
  - `cov`    float32 `[N,2,2]` — RaCo keypoint covariance in pixels² (usable as a BA weight); `NULL` for ALIKED
- `desc` is optional too (`D = 0` and `NULL` for a detector without descriptors).
- `xy` means the same for every model: `(0,0)` is the centre of the top-left pixel. The original RaCo adds 0.5; we do not.
- `score` is model-specific: ALIKED in [0, 1]; RaCo a probability that sums to 1 over the image (~1e-4 for a good keypoint).
- `config.model` is the detector (`aliked-n16` or `raco`); `config.descriptor` optionally names a second model
  that describes its keypoints: `model = raco`, `descriptor = aliked-n16`. It has its own weights source.
  `skip_ranker` / `skip_covariance` leave out a RaCo head and its output.
- RaCo selection is always "best `top_k`" (default 2048); `score_threshold` and `max_keypoints` do not apply.
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
- [ ] **M3 — Runs on the Mac GPU** (phase 4, FP32)
  Metal backend passes the same comparison on the M5 laptop.
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
- **RaCo** is no longer a last milestone; it has its own track below and its demo part moves into M7.

### RaCo track

Runs in parallel with M3-M8. Each step needs only the backend it names, so it never waits for a later milestone.

- [x] **R1 — RaCo reference locked** (same method as M1)
  Done: upstream pinned at `ee06fa6`, checkpoint `v1.0.0/raco.pth` (sha256 in `tools/fetch_upstream.sh`);
  `weights/raco.jkpw` is 93 tensors, 496,662 values, 1.99 MB. The reference forward pass from that file
  (`tools/raco_folded.py`) matches the original on 53 images (320 px to 1536x1024): keypoint overlap >= 99.95 %,
  position error <= 1.3e-4 px, score error <= 2.2e-5 of the best score, ranker error <= 5.4e-6 of the largest
  score, covariance error <= 1.4e-4 (relative).
- [x] **R2 — RaCo on the CPU backend**
  `jkp extract image.png --model raco` returns keypoints, scores, ranker scores and covariances.
  Done: the C++ output matches the original on the same 53 images with the same bounds (ranker 6.1e-6,
  score 2.7e-5); every intermediate tensor matches the golden dump within 1.3e-6 (relative).
  Speed is not a goal here: 0.42 s for 640x470 against 0.075 s for ALIKED on the same machine.
- [x] **R3 — RaCo keypoints with ALIKED descriptors** (CPU; any backend that has both afterwards)
  `jkp extract image.png --model raco --descriptor aliked-n16`; switches to skip the ranker and covariance heads.
  Done: on the same 53 images the descriptors match the original pairing (original RaCo keypoints described
  by the original ALIKED, as `ALIKED.describe` in cvg/LightGlue does it): descriptor cosine >= 1 - 3.4e-9,
  the other bounds unchanged. CPU times for 640x470: 0.46 s RaCo alone, 0.53 s with descriptors,
  0.125 s without the covariance head, 0.055 s without covariance and ranker heads.
- [ ] **R4 — RaCo on CUDA** (NVIDIA machine)
- [ ] **R5 — RaCo on Metal** (after M3)
- [ ] **R6 — RaCo on WebGPU** (with M6); detector switch, covariance ellipses and top-K by ranker score in the M7 demo.

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

### 8 — RaCo detector (RaCo track, R1-R6)
- Done (R1, R2): export `raco.pth` to `.jkpw`, reference pass, `compare.py --model raco`, `src/models/raco/` graph,
  CPU backend, golden test.
- Done (R3): `Aliked::describe_at` runs the ALIKED dense network without its score head and calls the existing
  `Ops::describe`, so it needs nothing new from a backend. The two networks share nothing but the input image,
  so the cost is the sum of both.
- R4-R6, per GPU backend: three operations (`normalize`, `pad_reflect`, `detect_raco`) and the LeakyReLU
  activation; everything else is already there for ALIKED. Then `raco_golden_test` and `compare.py` as on the CPU.
  Fusing reflect padding into the convolution and running the covariance head in FP16 belong to M5.
- **Exit:** RaCo keypoints, ranker scores and covariances match the original on every backend; RaCo + ALIKED
  descriptors selectable in the demo.

## RaCo

Source: https://github.com/cvg/RaCo (3DV 2026, Apache-2.0), reviewed at commit `ee06fa6`.

- **Detector only.** Outputs keypoints, detection scores, ranker scores and 2x2 covariances. No descriptors:
  the authors match with ALIKED descriptors + LightGlue. For this repo that means running the ALIKED
  network as well to describe RaCo keypoints — roughly two networks per image.
- **Backbone** is ALIKED-n16 widths (16/32/64/128) with plain 3x3 convolutions, no deformable convolution.
  Same pooling, upsampling, SELU and score head as ALIKED, so most kernels are reused.
- **New ops:** reflect padding, 5x5 convolution, LeakyReLU, Softplus, global softmax over the score map,
  ImageNet mean/std input normalization.
- **Ranker head** runs on the normalized full-resolution image: 9 residual blocks of 12 channels, then a 5x5
  convolution with reflect padding. About a fifth of the arithmetic.
- **Covariance head:** three 3x3 convolutions with reflect padding (128 -> 64 -> 32 -> 32, LeakyReLU) and a 1x1
  convolution on the aggregated features, at full resolution: 3 Cholesky elements per pixel, softplus on the
  diagonal ones, sampled at keypoints. This is the expensive part, about three quarters of the arithmetic
  of the whole network (and 73 % of the measured CPU time); `skip_covariance` leaves it out.
- **Detection:** softmax over the whole image, NMS in a 3x3 window (upstream calls the window size
  `nms_radius = 3`), best K (default 2048), subpixel refinement by a softmax over the 3x3 logits
  (temperature 0.5). Score, ranker and Cholesky values are bilinear samples at the refined position.
- **Conventions we fixed** (the reference pass and the C++ code agree on them):
  - keypoints are returned without upstream's +0.5, so they mean the same as ALIKED keypoints;
  - equal scores are ordered in raster order (upstream leaves ties to `torch.topk`);
  - suppressed pixels are never returned: an image with fewer than K local maxima gives fewer keypoints
    (upstream pads its output with arbitrary pixels).
- Weights are a release asset (`v1.0.0/raco.pth`) of the Apache-2.0 repository, which states one license for
  the project and none separately for the weights. The exported `weights/raco.jkpw` is committed with
  `weights/LICENSE-RACO` on that basis; ask the authors before a public release if this must be certain.

### Notes from R1 and R2

- `tools/raco_folded.py` is the executable specification, `tools/raco_upstream.py` runs the original and
  checks its own staged copy of the forward pass against it bit for bit.
- Value ranges differ a lot from ALIKED: probabilities are ~1e-9..1e-3 and ranker scores reach +-2000, so
  `compare.py` and `raco_golden_test` use relative errors for RaCo.
- Keypoints are ordered by score, and scores that differ only by float32 rounding swap places or fall on the
  other side of the top-K cut (at most 1 of 2048 in the test images). Tests pair keypoints by position, not by index.
- New backend operations have default implementations that throw "not implemented on this backend yet",
  so adding RaCo did not touch the CUDA backend or the Metal work in progress. `normalize` is the first
  operation of the graph, so an unsupported backend fails before doing any work.
- Reflect-padded convolutions are written as `pad_reflect` + ordinary convolution + crop; the crop is merged
  with the un-padding crop where both follow each other. Simple and exact, one extra copy per convolution.
- Softplus is applied to the four neighbours of each keypoint inside `detect_raco`, not to the whole map;
  the result is the same and no per-channel activation is needed.
- `JKP_EMBED_MODELS` accepted a list but embedded only its first entry; fixed. The default is still
  `aliked-n16` only (build with `-DJKP_EMBED_MODELS="aliked-n16;raco"` or pass `--weights weights/raco.jkpw`).
  Make `raco` part of the default once M3 has landed; that line of `CMakeLists.txt` sits next to the Metal options.
- R3: RaCo positions go to the ALIKED descriptor head as `2 * xy / (size - 1) - 1`, the formula of
  `ALIKED.describe` in cvg/LightGlue (commit `5a650c4`) once upstream's +0.5 is removed. The head truncates the
  pixel position it recovers, so keep this float32 order on GPU backends. Positions can lie up to one pixel
  outside the image; the head already treats everything outside as zeros.
- Golden files: `python tools/dump_golden.py --model raco <images>` writes `build/golden/raco/`; `ctest` picks them up.

## Development machines

- Phases 0, 1, 4, 5, 6, 7 run on the M5 laptop (WebGPU via Dawn uses Metal there). So do R1-R3, R5 and R6.
- Phases 2 and 3 and R4 need an NVIDIA GPU machine.
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

## Open questions (decide during implementation)

- Tensor layout for speed: NCHW vs NHWC on CUDA; texture vs storage-buffer representation in WGSL.
- WebGPU limits (storage buffer size, bindings per shader stage) at large resolutions — may need tiling.
- Whether to add `aliked-t16` as a second, faster web-friendly variant (same op set, different widths).
- Descriptor output quantization for the downstream matcher.
