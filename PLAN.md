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
- [x] **M5 — Fast** (phase 3, plus FP16 on Metal)
  Benchmark table: jkp CUDA vs PyTorch CUDA, jkp Metal, FP32 vs FP16, at three resolutions. FP16 accuracy loss measured.
  Done. Milliseconds per image, from an 8-bit image in host memory to keypoints and descriptors in host
  memory; top-k 1000, median of warm runs, columns timed in turn (`tools/bench.py`):

  | Size | Metal FP32 | Metal FP16 | PyTorch MPS | CUDA FP32 | CUDA FP16 | PyTorch CUDA |
  |---|---|---|---|---|---|---|
  | 640x480 | 8.3 | 7.2 | 13.1 | 15.2 | 13.4 | 15.6 |
  | 1241x376 | 10.3 | 8.7 | 19.2 | 19.6 | 17.5 | 22.2 |
  | 1024x768 | 13.8 | 11.3 | 28.9 | 26.8 | 24.5 | 31.8 |
  | 1920x1080 | 30.2 | 23.3 | 74.8 | 59.1 | 53.9 | 77.3 |

  Metal and MPS: M5 Pro, macOS 26. CUDA: RTX 2070 Max-Q laptop, CUDA 13.4, PyTorch 2.15 (cu132). PyTorch is
  the original ALIKED, without its compiled `get_patches` op (see M1 notes). Before M5: Metal 32.8 / 212 ms and
  CUDA 48.7 / 270 ms at 640x480 / 1080p. CPU backend for scale: 66 ms (M5 Pro) and 355 ms (laptop) at 640x480.
  - FP32 results did not change: `jkp extract` output is byte-identical to the M3 (Metal) and M4 (CUDA) builds
    on 58 images in threshold and top-k mode, so the comparison with the original stands as recorded there.
  - FP16 (`--precision fp16`, `config.precision`) against FP32 on the same backend, 24 images on Metal and 44
    on CUDA, threshold and top-k 1000 (`precision_test`): 98.7-100 % of the keypoints are found within 1 px in
    the other run (mean 99.6-99.9 %); median shift of those <= 0.002 px; score difference <= 0.04; descriptor
    cosine 1 - 2e-5 on average, >= 0.98 for the worst pair (neighbouring maxima almost 1 px apart).
    Against the original on 57 images (Metal, `compare.py`): keypoint overlap >= 99.45 %, mean 99.8 %.
  - Not reached: the CUDA backend is only level with PyTorch at 640x480 (25 % faster at 1080p). Its
    descriptor head takes about 6 of the 15 ms there; see the notes from M5.

  For reference, the ALIKED paper reports 77.4 FPS (12.9 ms) for aliked-n16 at 640x480 with 1K keypoints
  (PyTorch, RTX 2060).
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
  (M5 did FP16 storage with float32 math, tiled kernels and the lazy pyramid; pinned upload and batches were not needed for the exit.)
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

## Notes from M5

- Profilers: `JKP_METAL_PROFILE=1` gives each dispatch its own command buffer and prints GPU time per
  operation and per kernel; `JKP_CUDA_PROFILE=1` synchronizes after every launch and prints time per
  operation. Both print when the backend is destroyed. Summed times match normal wall time on both, so the
  kernels are the cost, not synchronization or encoding.
- Tiled convolutions. One thread per output value re-read the whole input once per output channel.
  `k_conv2d_tile` and `k_deform_sum_tile` compute T output channels per thread (4, 8 or 16 by output channel
  count; 32 is slower; `JKP_METAL_TILE` / `JKP_CUDA_TILE` force one). Weights are stored tap-major,
  `w[tap * cpad + oc]`, so the T weights of a tap are adjacent; Metal reads them four at a time through the
  constant address space. Each output keeps its accumulation order, so results are unchanged.
- The pyramid. `upsample_concat` and `l2_normalize_crop` are fused operations in `Ops` (defaults run the
  separate operations; the CPU backend uses those). On the GPU backends their result is lazy: the 128-channel
  full-resolution tensor (1 GB at 1080p) is never stored. The score head's 1x1 convolution computes the
  upsampled values per pixel from the four small maps, and the descriptor head normalizes only the pixels the
  keypoints touch. `Tensor::read` and any other operation store it on demand, which is what the golden tests
  check; stored or not, FP32 values are the same. This removed about a third of the 1080p time.
- FP16 design. `real` (Metal) / the template parameter `R` (CUDA) is the storage type of feature maps and of
  the dense network's weights; every value is converted to float when read, so arithmetic is float32 in
  both modes and one kernel source serves both. In FP16 mode the convolutions use a fused multiply-add.
  Kept in float32: the score map (half would make neighbouring scores tie in NMS), keypoint selection, and
  the descriptor head's weights and buffers (per keypoint, so it costs nothing). `src/core/half.h` converts
  on the host (checked against the compiler's `_Float16` on 400 M values).
- FP16 gains 13-23 % on Metal and 8-12 % on CUDA: after tiling the convolutions are no longer limited by
  memory traffic. Half arithmetic was not tried; it would need its own accuracy measurement.
- Top-k selection takes 16 bits of the float per radix pass instead of 8: two host round trips instead of four.
- CUDA descriptor head (about 6 ms for 1000 keypoints at any image size, against about 2 ms on Metal):
  one output per thread is faster there than tiling (`k_sddh_sf_tile<R, 1>`), the opposite of Metal. The
  rest is reading 128 channel planes per sampled pixel. Tried and dropped: storing the four small maps
  channel-last (descriptor sampling 3.0 -> 1.6 ms, but the dense kernels that read them got slower, 1080p
  57 -> 68 ms; no change on Metal), and gathering all sampled pixels with one thread per value (slower:
  the scattered reads dominate, not the thread count). Untried: channel-last copies of only the three
  low-resolution maps for the descriptor kernels.
- Still open, by GPU time: on Metal at 1080p the 3x3 convolutions at full and half resolution (about 9 of
  30 ms); on CUDA the descriptor head, then the same convolutions. Keypoints still make a host round trip
  between `detect` and `describe`.
- `tests/precision_test.cpp` (ctest `<backend>_fp16_vs_fp32`) fails on a broken FP16 path; its bounds are
  far looser than the measured values. `jkp bench` skips two warm-up runs (`--warmup`).
- The CUDA laptop has a separate tree at `~/Repos/just-keypoints/.claude/worktrees/m5` with its own `.venv`
  holding CUDA-enabled PyTorch; the M4 worktree's `.venv` is CPU-only.

## Open questions (decide during implementation)

- Tensor layout for speed: NCHW vs NHWC on CUDA; texture vs storage-buffer representation in WGSL.
- WebGPU limits (storage buffer size, bindings per shader stage) at large resolutions — may need tiling.
- Whether to add `aliked-t16` as a second, faster web-friendly variant (same op set, different widths).
- Descriptor output quantization for the downstream matcher.
