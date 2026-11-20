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
| Second detector | RaCo (cvg/RaCo, Apache-2.0), developed alongside the backends instead of after them: it runs on the CPU backend now and reaches each GPU backend once that backend passes ALIKED. See "RaCo track" and "RaCo" below. A third, XFeat (verlab/accelerated_features, Apache-2.0), for light and mobile use: see "XFeat track". |
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
web/                     WebAssembly entry points, JS wrapper (npm package in M8), browser test page
docs/                    documentation site + live demo (static, Cloudflare Pages)
bindings/python/         the Python package just_keypoints (nanobind over the C API), its PyPI README and tests
packaging/               release scripts: the version, the CUDA package variants, the CUDA toolkit download
```

PyTorch is used only in `tools/` to produce weights and test vectors. It is never a runtime dependency. The
Python package (`bindings/python`) is a thin binding of the C API and needs only NumPy.

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
- [x] **M6 — Runs in the browser** (phase 5)
  WebGPU backend passes the comparison natively (Dawn) and in a browser test page.
  Done (M5 Pro, macOS 26, Dawn v20261113.214538 natively, Chrome 156 in the desktop app's browser pane):
  - Natively, `jkp extract --backend webgpu` matches the original on 57/57 sample images in threshold and
    top-k mode: identical keypoint sets, position error <= 2.4e-4 px, score error <= 7.6e-5, descriptor cosine
    >= 1 - 3.6e-9 (Metal in M3: 2.6e-4, 8e-5, 1 - 4e-9). Every intermediate tensor matches the golden dump
    within 2.3e-5; all 24 WebGPU tests pass (operations, against the CPU, 20 golden frames, FP16 against FP32,
    the caller's device).
  - In Chrome (`web/test.html`, 10 images from 640x480 to 2075x1491, threshold and top-k 1000): WebGPU FP32 gives
    the native CPU backend's keypoint sets on every image (position <= 1.4e-4 px, score <= 4.7e-5, cosine
    >= 1 - 6e-7); FP16 finds 99.6-100 % of them within 1 px. The module's own CPU backend agrees as closely.
  - FP16 against the original (57 images, threshold): overlap >= 99.31 %, mean 99.77 % (Metal: 99.45 %, 99.8 %).
  - Speed, top-k 1000, ms per image (`tools/bench.py`, same method as M5):

    | Size | Metal FP32 | WebGPU FP32 | Metal FP16 | WebGPU FP16 |
    |---|---|---|---|---|
    | 640x480 | 8.4 | 10.1 | 7.3 | 8.8 |
    | 1241x376 | 10.6 | 12.7 | 8.7 | 11.0 |
    | 1024x768 | 13.9 | 17.1 | 11.2 | 15.2 |
    | 1920x1080 | 29.7 | 38.0 | 23.2 | 33.3 |

    Native WebGPU takes 1.2-1.4 times Metal's time. In Chrome, median of 5 warm runs with the pane hidden:
    12-13 ms at 640x480 and 58 ms at 2075x1491 in FP32, 10 ms at 640x480 in FP16. The CPU backend in the
    browser (one thread) takes about 1.2 s at 640x480.
  - Same keypoint sets, not always the same order: keypoints whose scores nearly tie can swap in top-k mode
    and under the cap. See the notes from M6.
- [x] **M7 — Public demo and docs** (phase 6)
  Cloudflare Pages site (as just-ba's): load two images, see keypoints, matches and timings.
  Done: published at <https://just-keypoints.pages.dev> by the `docs` workflow (`deploy=true`), last from `main`
  at `b82a474` (R4), so the live demo has RaCo (R6). Checked on the live site (M5 Pro, Chromium 156 in the
  desktop app's browser pane, in front): TUM pair, top-k 1000, WebGPU FP32: ALIKED 229 matches, 10.7 / 12.9 ms;
  RaCo 318 matches, 55.7 ms per image; benchmark page 11.6 / 15.1 / 19.3 / 40.8 ms FP32 and 10.9 / 12.2 /
  17.8 / 35.7 ms FP16. The fallback without WebGPU, also on the live site, in the notes from M7.
  - Site: Sphinx + MyST in `docs/`, just-ba's theme with this project's accent; pages: build, quick start, in the
    browser, accuracy and speed, C API, weights format, references. `sphinx-build -W` passes.
  - Live demo (`docs/_extra/demo/`): two sample pairs (TUM RGB-D, KITTI), upload, drop or camera; keypoints and
    match lines; best-K or threshold, resolution, FP32/FP16, WebGPU/CPU, ratio test. TUM pair, top-k 1000,
    WebGPU FP32: 11-13 ms per image, 229 matches at ratio 0.9; KITTI: 14-16 ms FP32, 12-13 ms FP16.
    The CPU backend in the page: 0.5-0.8 s per image at 640x194.
  - Benchmark page (`docs/_extra/bench/`): the native benchmark's four sizes, top-k 1000, median of warm runs,
    copied out as Markdown. With the pane hidden: 13.6 / 16.5 / 23.0 / 41.2 ms in FP32 and 12.0 / 15.1 / 20.2 /
    36.0 ms in FP16 (640x480, 1241x376, 1024x768, 1920x1080).
  - Not as planned: the timings are per step of the page (prepare, extract, match), not per stage inside the
    library; see the notes from M7.
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
- [x] **R4 — RaCo on CUDA** (NVIDIA machine)
  `jkp extract image.png --model raco --backend cuda`, with `--descriptor aliked-n16` and the head switches.
  Done (RTX 2070 Max-Q laptop, CUDA 13.4, driver 620.70), on the R1 image set (53 images), against the original
  (PyTorch on the CPU): keypoint overlap >= 99.95 %, position error <= 2.4e-4 px, score error <= 5.6e-5 of the
  best score, ranker error <= 6.3e-6 of the largest score, covariance error <= 1.4e-4 (relative); with ALIKED
  descriptors, descriptor cosine >= 1 - 1.5e-9 and the other bounds unchanged (Metal in R5: 1.2e-4 px, 4.3e-5,
  7.5e-6, 1.4e-4, 1 - 1.3e-9). The position bound is one float32 step at x > 2048 (`imw.png`, 2522 px wide);
  every other image is within 1.2e-4 px. Every intermediate tensor of the three RaCo golden files matches within
  1.3e-6 (relative); `cuda_raco_vs_cpu` pairs every keypoint one to one on all 13 cases (24,646 keypoints on the
  largest), every stage within 2.0e-6; `ops_test` passes the RaCo operations in FP32 and FP16 storage.
  FP32 only, as on the other backends. Not tuned: 89 ms at 640x480 and 626 ms at 1920x1080, 6-10 times ALIKED
  on the same GPU (table in R6). See the notes from R4.
- [x] **R5 — RaCo on Metal** (after M3)
  `jkp extract image.png --model raco --backend metal`, with `--descriptor aliked-n16` and the head switches.
  Done (M5 Pro, macOS 26), on the R1 image set (53 images, 501x380 to 2522x818), against the original:
  keypoint overlap >= 99.95 %, position error <= 1.2e-4 px, score error <= 4.3e-5 of the best score, ranker
  error <= 7.5e-6 of the largest score, covariance error <= 1.4e-4 (relative); with ALIKED descriptors,
  descriptor cosine >= 1 - 1.3e-9 and the other bounds unchanged. The CPU backend in R2/R3: 2.7e-5 score,
  6.1e-6 ranker, 1 - 3.4e-9 cosine. Every intermediate tensor of the three RaCo golden files matches within
  1.3e-6 (relative); against the CPU backend on generated images (`metal_raco_vs_cpu`) every stage is within
  2.3e-6 and the keypoint sets are identical.
  FP32 only, like the CPU backend (the API refuses `precision = fp16` for RaCo; see the notes from R5).
  Not tuned. Timed with `tools/bench.py` in R6 (table there): 44.6 ms at 640x480 and 286 ms at 1920x1080 on
  Metal, 5-10 times ALIKED. The covariance head is about 60 % of a run, the ranker head most of the rest;
  without both heads RaCo takes about as long as ALIKED.
- [x] **R6 — RaCo on WebGPU** (with M6); detector switch, covariance ellipses and top-K by ranker score in the M7 demo.
  `jkp extract image.png --model raco --backend webgpu`, with `--descriptor aliked-n16` and the head switches;
  in the browser `jkp.create({ weightsUrl: 'raco.jkpw', descriptorWeightsUrl: 'aliked-n16.jkpw' })`.
  Done (M5 Pro, macOS 26, Dawn v20261113.214538 natively, Chrome 156 in the desktop app's browser pane):
  - Natively, on the R1 image set (53 images), against the original: keypoint overlap >= 99.95 %, position error
    <= 1.2e-4 px, score error <= 9.0e-5 of the best score, ranker error <= 1.2e-5 of the largest score, covariance
    error <= 1.6e-4 (relative); with ALIKED descriptors, descriptor cosine >= 1 - 1.7e-9 and the other bounds
    unchanged (Metal in R5: 4.3e-5, 7.5e-6, 1.4e-4, 1 - 1.3e-9). Every intermediate tensor of the three RaCo
    golden files matches within 1.3e-6 (relative); `webgpu_raco_vs_cpu` pairs every keypoint one to one on all
    12 cases (24,646 keypoints on the largest); `ops_test` passes the RaCo operations in FP32 and FP16 storage.
  - In Chrome (`web/test.html`, the 10 images of M6, top 2048 with ALIKED descriptors, against the native CPU
    backend): WebGPU pairs 2048 / 2048 keypoints on every image, position <= 6.8e-5 px, score <= 6.6e-5,
    ranker <= 1.1e-5, covariance <= 1.5e-4, descriptor cosine >= 1 - 5.0e-7. The module's CPU backend
    agrees as closely (9 images; the 3-megapixel one does not fit its memory).
  - Speed, ms per image, same method as M5 (`tools/bench.py`: top-k 1000, FP32, median of 30 warm runs, median
    over 3 rounds, columns timed in turn; load average below 2, and ALIKED on Metal reproduces M5's numbers). CPU
    backend: 3 runs, one round, all threads. CUDA: RTX 2070 Max-Q laptop; its ALIKED row is M5's, and R4
    reproduced it within 3 % (15.3 / 19.9 / 27.4 / 60.8 ms) in the same run as its RaCo rows.

    | Size | Model | CPU | Metal | WebGPU | CUDA |
    |---|---|---|---|---|---|
    | 640x480 | ALIKED | 68 | 8.3 | 9.7 | 15.2 |
    | | RaCo | 410 | 44.6 | 45.8 | 89.1 |
    | | RaCo + ALIKED descriptors | 465 | 51.2 | 52.7 | 103.2 |
    | | RaCo without covariance | 109 | 18.3 | 19.0 | 30.9 |
    | 1241x376 | ALIKED | 101 | 10.3 | 12.4 | 19.6 |
    | | RaCo | 649 | 67.4 | 69.6 | 139.6 |
    | | RaCo + ALIKED descriptors | 730 | 75.4 | 78.4 | 157.0 |
    | | RaCo without covariance | 167 | 26.0 | 27.6 | 47.2 |
    | 1024x768 | ALIKED | 164 | 13.7 | 17.2 | 26.8 |
    | | RaCo | 1080 | 109.0 | 112.7 | 231.6 |
    | | RaCo + ALIKED descriptors | 1211 | 119.7 | 124.6 | 254.4 |
    | | RaCo without covariance | 287 | 41.9 | 43.3 | 77.6 |
    | 1920x1080 | ALIKED | 376 | 29.5 | 37.7 | 59.1 |
    | | RaCo | 2732 | 286.1 | 295.2 | 626.3 |
    | | RaCo + ALIKED descriptors | 3050 | 308.9 | 319.0 | 675.6 |
    | | RaCo without covariance | 750 | 107.6 | 111.0 | 207.7 |

    RaCo costs about 5 times ALIKED at 640x480 on the GPUs and 8-10 times at 1920x1080: its heads run at full
    resolution (CUDA: 6 and 10 times). WebGPU is within 2-4 % of Metal on RaCo (15-28 % slower on ALIKED). An earlier run at 640x480
    with 2048 keypoints, under load: without both heads 8.3 ms on WebGPU and 8.1 ms on Metal. In Chrome with
    descriptors, top 2048, pane hidden: 56 ms at 640x480, 81 ms at 1241x376, 0.47 s at 2075x1491. The module's
    CPU backend (one thread) takes about 12 s at 640x480, against 1.1 s for ALIKED.
  - Demo: detector switch (ALIKED, or RaCo described by ALIKED), best K by ranker score, covariance ellipses at
    3, 10 or 30 sigma. TUM pair, WebGPU: RaCo top 1000 gives 318 matches at ratio 0.9 (ALIKED: 229); 2000
    detected and the 500 best by ranker kept give 224 matches.

### XFeat track

XFeat (CVPR 2024, verlab/accelerated_features, Apache-2.0): a detector and 64-dim descriptor at about a third of
ALIKED's time on a CPU and half on a GPU, for the mobile demo of like-recon and for quick tests. Sparse mode only (`detectAndCompute`);
the semi-dense XFeat* matches per pair, so it has no per-image keypoints and does not fit a feature library.
"Both versions" are the two precisions: FP32 (reference, default) and FP16 on the GPU backends, as for ALIKED;
upstream ships one FP32 checkpoint (`weights/xfeat.pt`), FP16 converts at load time.

- [x] **X1 — Reference locked:** upstream pinned at `e92685f` (`weights/xfeat.pt`, sha256 in
  `tools/fetch_upstream.sh`); `weights/xfeat.jkpw` is 54 tensors, 659,282 values, 2.64 MB (BatchNorm folded, the
  fine matcher left out). `tools/xfeat_folded.py` matches the original on 59 images (ALIKED's and XFeat's assets,
  320 px to 2522x818): identical keypoint sets and positions, score error <= 4.5e-6, descriptor cosine
  >= 1 - 3.2e-10, also at a top-k cut of 1000. `compare.py --model xfeat`, `dump_golden.py --model xfeat`.
- [x] **X2 — CPU backend:** `jkp extract --model xfeat`. On the same 59 images: identical keypoint sets and
  positions, score error <= 2.7e-5, descriptor cosine >= 1 - 5.4e-11; every intermediate tensor of the golden
  files within 2.2e-6 (relative). 22 ms at 640x480 (ALIKED: 68 ms).
- [x] **X3 — Metal**, FP32 and FP16. FP32 against the original: identical sets and positions, score <= 2.7e-5,
  cosine >= 1 - 1.9e-10. FP16: overlap >= 99.0 % (mean 99.4 %), cosine >= 1 - 2.9e-4.
- [x] **X4 — WebGPU**, natively and in the browser. Natively: identical sets and positions, score <= 5.3e-6,
  cosine >= 1 - 1.1e-10; FP16 as Metal. In Chrome (the demo, TUM pair, top 1000): 244 matches at ratio 0.9
  (ALIKED: 229), 12-17 ms per image; the module's CPU backend 0.36-0.42 s.
- [ ] **X5 — CUDA** (NVIDIA machine): the six operations, as on Metal. Until then XFeat on CUDA throws "XFeat is
  not implemented on this backend yet"; callers fall back to the CPU (22 ms at 640x480 on the M5 Pro).
- [x] **X6 — Bindings and docs:** Python (`model="xfeat"`, tests), browser module (`xfeat.jkpw` is copied
  beside it), demo detector switch, README, references, accuracy and speed.

Speed, ms per image, top-k 1000, median of 20 warm runs over 3 rounds (`tools/bench.py --model xfeat`, M5 Pro):

| Size | CPU | Metal FP32 | Metal FP16 | WebGPU FP32 | WebGPU FP16 | PyTorch MPS | PyTorch CPU |
|---|---|---|---|---|---|---|---|
| 640x480 | 21.9 | 3.9 | 3.3 | 4.0 | 3.9 | 4.3 | 23.8 |
| 1241x376 | 28.2 | 4.3 | 3.6 | 4.4 | 4.2 | 4.3 | 30.2 |
| 1024x768 | 46.0 | 5.5 | 4.4 | 5.8 | 5.4 | 5.5 | 57.0 |
| 1920x1080 | 98.4 | 10.6 | 8.2 | 12.1 | 11.2 | 9.9 | 149.7 |

Not tuned: the 64-channel 3x3 convolutions at 1/8 resolution are a third of the GPU time.

New operations (everything else is ALIKED's): ReLU, 3x3 convolution with stride 2, instance norm over the image,
bilinear resize with `align_corners = false` to a given size, space-to-depth by 8, and `detect_xfeat`: softmax
over 65 channels, depth-to-space, 5x5 NMS above 0.05, score = heatmap x bilinear reliability, best `top_k`
(default 4096), descriptors by bicubic sampling of the L2-normalized 64-channel map, normalized again.

Conventions fixed (the reference pass and the C++ code agree on them):
- the image is in [0, 1] and is resized (not padded) to a multiple of 32, as upstream;
- keypoints mean what ALIKED's mean, `(0, 0)` the centre of the top-left pixel: `(i + 0.5) * r - 0.5` for the
  resized image's pixel `i`, where upstream returns `i * r` (at most 0.5 (r - 1) px apart);
- equal scores are ordered in raster order;
- kept from the original, because they decide which keypoints survive: positions are normalized by
  `W - 1` and sampled with `align_corners = false`, so the nearest-neighbour read of the heatmap falls outside the
  map on the last row and column (score 0, dropped), and a keypoint at pixel (0, 0) is dropped (upstream marks
  its padding that way). WGSL divides exactly there (`div_rn`): an approximate 1 would keep them.
- FP16 keeps the 65 keypoint logits in float, like ALIKED's score map, so the softmax does not tie neighbours.

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
- FP32 reference mode; opt-in FP16 as storage only, arithmetic float32 as on Metal and CUDA. Storage is
  `array<f16>`, so it needs the `shader-f16` feature (`jkp_create` fails without it). Packing two values per
  32-bit word with `pack2x16float` would need no feature, but neighbouring threads write the two halves of
  one word, so every kernel would have to write pairs.
- **Exit:** minimal test page; results match golden within FP16 tolerance.

Settled before starting (2026-11-15), so that just-keypoints, just-ba and just-matching can be linked into
one app. just-ba already ships a WebGPU backend built this way (its document 16, decisions D120-D128):

- **Same toolchain as just-ba:** Dawn's prebuilt release v20261113.214538 natively, emdawnwebgpu from the
  same release and Emscripten 7.0.0 in the browser, unpacked under `third_party/webgpu` (ignored by git;
  `JKP_DAWN_DIR`, `JKP_EMDAWNWEBGPU_DIR`). Dawn's `webgpu.h` changes between releases; an app links one Dawn.
  Rejected: wgpu-native (its `webgpu.h` differs from emdawnwebgpu's), building Dawn from source.
- **Same WebAssembly settings as just-ba:** single-threaded host (no `SharedArrayBuffer`, any static host),
  JSPI for waits on the GPU, native wasm exceptions. Asyncify cannot be combined with native exceptions.
- **The library is a static library under Emscripten too.** The `.js`/`.wasm` module and the JS wrapper are
  a separate target. An app links all three libraries into one module: separate modules cannot share a
  GPU device through emdawnwebgpu.
- **The caller may supply the device** (`jkp_config.webgpu_device`, a `WGPUDevice`); otherwise the backend
  creates its own. Dawn's device is not thread-safe, and three libraries each opening one would mean three
  memory budgets next to FlexGlue's ~1 GB.
- **Selection on the device, as on CUDA.** Metal's host-side prefix sums rely on unified memory; on WebGPU
  every readback is a map and a wait, slower still in the browser.
- **One binding per tensor.** Kernels bind at most a handful of buffers; the device requests the adapter's
  `maxStorageBufferBindingSize` and `maxBufferSize` (the defaults are 128 and 256 MiB).
- **CPU fallback in the browser runs on one thread** (`std::thread` needs pthreads).
- **WGSL cannot turn off fused multiply-add**, so WebGPU is not expected to reproduce the CPU's keypoint
  sets bit for bit as Metal and CUDA do. Measured against the original with the overlap metrics of M5.

### 6 — Documentation site and live demo
- Static site in `docs/`, built by a workflow started by hand and deployed to Cloudflare Pages (as just-ba's). No server: wasm, shaders and weights are static assets.
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
- Done: Python package on PyPI, built as just-ba's. nanobind (stable ABI, one wheel per platform for Python
  3.12 and later) rather than the ctypes binding first planned: the same build as the sibling libraries, and
  the type stubs come from the module. `just_keypoints.Extractor(**config).extract(image)` returns a
  `Features` with NumPy copies of `xy`, `score`, `desc`, `rank` and `cov`; errors raise `RuntimeError` with
  `jkp_last_error()`. Both models' weights are built in (ALIKED BSD-3-Clause, RaCo Apache-2.0; both licence
  files ship in the wheel). Three packages provide the one module:

  | Package | Wheels | Backends |
  |---|---|---|
  | `just-keypoints` | Linux x86_64, macOS arm64 (13+), Windows x86_64, and the sdist | CPU; Metal on macOS |
  | `just-keypoints-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA 12.9 (sm_60 to sm_120, PTX for newer) |
  | `just-keypoints-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA 13.4 (sm_75 to sm_120, PTX for newer) |

  The CUDA runtime is linked statically (CMake's default), so the CUDA wheels depend on nothing from NVIDIA but
  the driver. WebGPU is in no wheel. The toolkit comes from NVIDIA's redistributable archives
  (`packaging/fetch_cuda.py`) and `packaging/variant.py` turns `pyproject.toml` into a CUDA package.
- **Release**: `.github/workflows/release.yml` runs on every push to main (a merge included) and publishes;
  pushes that change only the notes, `docs/`, `tools/` or `web/` release nothing. By hand,
  `gh workflow run release` builds and tests every wheel without publishing (`-f publish=true` publishes).
  The workflow chooses the version (`packaging/version.py`): the highest of the tags `vX.Y.Z` and the three
  packages' versions on PyPI with its patch bumped, or `project(VERSION)` in CMakeLists.txt when that is higher,
  so a minor or major release is an edit there in the change that warrants it; with nothing released yet,
  `project(VERSION)` itself (0.1.0). It is written into the checkout before each build and never committed; the
  tag `vX.Y.Z` is pushed once every package is up. Publishing is PyPI trusted publishing, with no GitHub
  environment and no token. A GPU change: run the GPU tests by hand before merging (`ctest` on the Mac and
  the CUDA machine, or `pytest bindings/python/tests` against a wheel from a by-hand run).
- **First release**: PyPI takes one pending publisher per repository and workflow at a time (as just-ba found),
  so the packages go up one at a time. Before merging, add the pending publisher for `just-keypoints` on PyPI
  (owner `manlito`, repository `just-keypoints`, workflow `release.yml`, no environment). The merge publishes it
  and the cu12 upload fails; add the pending publisher for `just-keypoints-cu12`, re-run the failed publish job
  (`gh run rerun <run> --failed`; uploaded files are skipped), then the same for `just-keypoints-cu13`. After
  that all three are ordinary trusted publishers and each release publishes them in one run.

### 8 — RaCo detector (RaCo track, R1-R6)
- Done (R1, R2): export `raco.pth` to `.jkpw`, reference pass, `compare.py --model raco`, `src/models/raco/` graph,
  CPU backend, golden test.
- Done (R3): `Aliked::describe_at` runs the ALIKED dense network without its score head and calls the existing
  `Ops::describe`, so it needs nothing new from a backend. The two networks share nothing but the input image,
  so the cost is the sum of both.
- R4-R6, per GPU backend: three operations (`normalize`, `pad_reflect`, `detect_raco`) and the LeakyReLU
  activation; everything else is already there for ALIKED. Then `raco_golden_test` and `compare.py` as on the CPU.
  Fusing reflect padding into the convolution and running the covariance head in FP16 belong to M5.
- Done (R5): Metal. The 5x5 ranker convolution got tiled instantiations (`k_conv2d_t*_k5`); the top-k
  selection of `detect` is shared with `detect_raco` (`Selection` in `metal_ops.mm`).
- Done (R6): WebGPU, the same port (`Selection` in `webgpu_ops.cpp`, `k_conv2d_tile` with K = 5); the browser
  module takes RaCo weights and descriptor weights, the demo has the detector switch.
- Done (R4): CUDA, the same port (`Selection` in `cuda_ops.cu`, `k_conv2d_tile` instantiated with K = 5).
- **Exit:** RaCo keypoints, ranker scores and covariances match the original on every backend; RaCo + ALIKED
  descriptors selectable in the demo. Met with R4 (CPU, CUDA, Metal, WebGPU).

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
- `JKP_EMBED_MODELS` accepted a list but embedded only its first entry; fixed. Since R5 the default is
  `aliked-n16;raco` (about 4.7 MB; `-DJKP_EMBED_MODELS=aliked-n16` leaves RaCo out). The WebAssembly build
  is configured with `-DJKP_EMBED_WEIGHTS=OFF` (README, docs workflow), so the site is not affected; CMake does
  not enforce that under Emscripten.
- R3: RaCo positions go to the ALIKED descriptor head as `2 * xy / (size - 1) - 1`, the formula of
  `ALIKED.describe` in cvg/LightGlue (commit `5a650c4`) once upstream's +0.5 is removed. The head truncates the
  pixel position it recovers, so keep this float32 order on GPU backends. Positions can lie up to one pixel
  outside the image; the head already treats everything outside as zeros.
- Golden files: `python tools/dump_golden.py --model raco <images>` writes `build/golden/raco/`; `ctest` picks them up.

### Notes from R5

- The softmax over the image takes five dispatches: block maxima and their maximum, block sums of
  `exp(l - max)` and their sum, then the probabilities. The CPU backend sums in double in raster order; Metal
  uses compensated (Kahan) float sums in blocks of 256, so probabilities agree to about 1e-7 (relative).
- NMS keeps a pixel when its probability equals the maximum of its window, computed on the probabilities
  rather than the logits, as on the CPU: two different logits can round to the same probability.
- Selection: count the nonzero NMS values (one round trip), then the radix select and rank sort of `detect`
  for the best `min(top_k, count)`; refinement and sampling run on the kept keypoints only.
- Metal has no `log1p`; softplus uses `log(y) * u / (y - 1)` with `u = exp(x)`, `y = 1 + u`, accurate to a few ulp.
- On large images (1024x768, 5000 keypoints) the keypoint sets are identical but the order of near-tied scores
  differs from the CPU's. So `backend_test` with `raco.jkpw` pairs keypoints by position, one to one: without a
  top-k cut every keypoint must pair, with one a keypoint may lack a partner only among the last 0.2 % of its list.
  Exact equality cannot be guaranteed: `exp` and the softmax sum differ by an ulp between the backends, and two
  distinct logits can round to the same probability on one backend only, which changes the NMS ties.
  Scores, ranker and covariances are bilinear samples at the refined position; they are compared at 1e-4.
- Probabilities below FLT_MIN (a logit about 87 or more below the maximum) are not maxima, on the CPU and on
  Metal: the GPU flushes such denormals to zero. Only matters when top_k exceeds the number of normal-range
  maxima on a map with that wide a logit range; real images have logits within about +-5.
- `k_rank_sort` is O(k^2), shared with ALIKED. RaCo has no score threshold, so a very large top_k (1e5 on a
  large image) would sort ~1e5 keypoints in one dispatch; the default 2048 is fine.
- `ops_test` checks the RaCo operations on backends that have them and reports "skipped" on the others.
- FP16: the operations work with half storage (`ops_test metal:fp16` passes), but the logits come from a
  convolution without an activation and are stored as half, so neighbouring logits tie and the set of local
  maxima changes (290 against 295 on the test map). Before lifting the API's FP32-only rule for RaCo, keep the
  logits in float (as ALIKED's score map) and measure the result against FP32.
- Speed is dominated by the full-resolution heads: the covariance head is 55-65 % of the run time and the
  ranker (nine residual blocks at full resolution, 5x5 output) most of the rest; neither is tiled or fused yet.

### Notes from R6

- The WebGPU kernels are the Metal ones (R5) in WGSL: softmax in five dispatches with compensated float sums,
  NMS on the probabilities with the FLT_MIN rule, softplus without `log1p`.
- `k_raco_refine` writes xy, scores, ranker scores and Cholesky elements into one buffer: WebGPU guarantees only
  eight storage buffers per kernel, and separate outputs would need nine.
- WGSL reserves `patch` as a keyword (a shader compile error names it); the refinement window is `win`.
- WebGPU differs from the CPU about three times as much as Metal does (against the CPU on `imw.png`: score
  6.4e-5 against 1.9e-5, ranker 1.2e-5 against 3.5e-6, covariance 1.2e-4 against 3.4e-5). The ranker and
  covariance heads do not go through the softmax, so the difference comes from the dense network: WGSL cannot
  turn off contraction into fused multiply-add (M6). The keypoint sets are unaffected.
- Speed is the covariance head's, as on Metal: about 27 of the 46 ms at 640x480 (59 %), 184 of 295 ms at
  1920x1080 (62 %); the ranker head most of the rest. WebGPU is level with Metal here because both spend that
  time in the same untuned full-resolution convolutions.
- Demo: RaCo's covariances are about a pixel (mean largest sigma 0.9 px on the TUM frames), so at 3 sigma an
  ellipse is about one screen pixel when a 640 px image is shown 250 px wide; the default is 10 sigma, and the
  page says so. Subsampling by ranker score runs in the page on the returned keypoints (sort by `rank`, keep
  K), as RaCo's README describes; the library still returns the best `top_k` by detection score.
- The browser module's `jkpw_create` takes the descriptor weights and the head switches; results add `rank` and
  `cov` (null when absent). `web/test.html` has a RaCo run per backend (FP32 only), with test data from
  `tools/web_testdata.py` (`--raco-weights`, on by default).

### Notes from R4

- The CUDA kernels follow the CPU backend; the file is compiled with `--fmad=false`, as for ALIKED. The softmax
  sum is in double (block sums of `expf(l - max)`, then their sum), closer to the CPU's raster-order double sum
  than the compensated float sums of Metal and WebGPU, which have no double. Softplus uses `log1pf` as on the CPU.
  NMS on the probabilities with the FLT_MIN rule of R5.
- `Selection` (count, ordered write, radix select + rank sort) is now shared by `detect` and `detect_raco`, as on
  Metal and WebGPU. ALIKED's results do not change: `jkp extract --backend cuda` output is byte-identical to the
  M5 build on the 53 images, in threshold and top-k mode, FP32 and FP16 (212 runs).
- 5x5 convolutions (the ranker output, 12 -> 1) use the tiled kernel; before R4 they would have fallen back to
  the one-value-per-thread `k_conv2d`. Same accumulation order, so the same results.
- `compare.py` against the original on the same GPU runs out of memory on the 3-megapixel images: PyTorch's
  caching allocator keeps its blocks in the same process. The R4 numbers use the original on the CPU
  (`--a upstream:cpu`). On its own, `nvidia-smi` shows at most 6.1 GB at 2522x818 and 7.4 GB at 2075x1491 (with
  ALIKED descriptors) on the 8 GB card. That includes blocks the pool keeps for reuse, which it frees and
  retries when an allocation fails, but full-resolution 128-channel tensors are 1.6 GB each at that size, so
  on an 8 GB card RaCo at 3 megapixels needs the GPU mostly to itself.
- Speed (`JKP_CUDA_PROFILE=1`, 640x480, 98 ms of kernel time): the covariance head is 64 % (its 128 -> 64
  convolution alone 43 ms, about 1 TFLOP/s without fused multiply-add), the ranker head 21 %, `detect_raco`
  about 1 ms. CUDA takes about twice Metal's time on RaCo, as on ALIKED in FP32 (15.2 against 8.3 ms): the
  same untuned full-resolution convolutions on a smaller GPU.

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

## Notes from M6

- `src/backend/webgpu/kernels.wgsl` is a kernel-by-kernel port of the Metal kernels (same names, arithmetic and
  order); `webgpu_ops.cpp` is the host side over `webgpu.h`, the same source natively (Dawn) and in the browser
  (emdawnwebgpu). The `Ops` interface did not change. WGSL has no templates: a kernel section holds `$T`, `$K`,
  `$O` and the host instantiates it by name (`k_conv2d_t8_k3_f32`); each instance is its own shader module, so
  sections declare their own bindings. Pipelines are created on first use.
- Arguments: buffers bind in the order they are passed; scalars go into a 256-byte uniform slot per dispatch
  (a ring written just before the submission), read in WGSL by position (`pi(0u)`, `pf(3u)`). Each dispatch has
  its own bind group. Offsets into a buffer (concatenation, compaction) are scalars, not binding offsets, which
  WebGPU requires to be multiples of 256 bytes.
- Ordering: one compute pass per batch, submitted when the host needs a value or writes a buffer. A
  `wgpuQueueWriteBuffer` runs before work submitted after it, even if encoded earlier, so every write submits
  the open batch first; with that, a buffer returned to the pool can be reused at once, as on Metal.
- Selection as on CUDA (count, scan and write on the device). Round trips per image: threshold mode one count
  and the results; top-k two histogram passes and the results; the descriptor head one per 8192 keypoints.
  Several ranges are copied into one staging buffer per round trip.
- Errors: every batch is wrapped in validation and out-of-memory error scopes, checked at the next read, so
  they are reported also on a device the caller created (`webgpu_shared_test`).
- WGSL gives no control over floating-point modes, and on Dawn's Metal path division is not correctly rounded
  (a correctly rounded replacement removed the flips below; whether Dawn asks for fast math was not checked)
  and multiply-adds may be fused. Values then differ from the CPU in the last bits (stages within
  2.3e-5, the same order of size as Metal). One place matters: a keypoint on an integer pixel recovered as
  4.9999995 truncates to the neighbouring 3x3 patch in the descriptor head (2 of 20 golden frames; descriptor
  values off by up to 0.017). `k_refine` therefore divides with `div_rn`, a correctly rounded division from two
  `fma` corrections; since then every golden test passes. Keypoints whose scores nearly tie can still come out
  in another order (top-k and the cap sort by score): same set, different order, natively and in Chrome alike.
- `tests/ops_test.cpp` (`<backend>_ops`) compares each operation with the CPU backend on random weights; it
  found the one host bug (the plain `k_conv2d` section taken for `k_conv2d_t8_k3`) at once. Any GPU backend
  can use it.
- Browser module: 293 KB of wasm and 100 KB of JS (`-sJSPI`, `-fwasm-exceptions`, `MODULARIZE`, `EXPORT_ES6`,
  memory growth to 4 GB), weights fetched separately (2.7 MB). `jkpw_create` and `jkpw_extract` are JSPI
  exports. `web/test.html` runs WebGPU FP32 and FP16 and the module's CPU backend on images decoded natively
  (raw RGBA from `tools/web_testdata.py`), against the native CPU backend's results; `window.jkpTest` resolves
  to the summary, so the check can be automated.
- The CPU backend in WebAssembly: one thread, about 1.2 s at 640x480 (native on all cores of the M5 Pro: 66 ms), and
  at most about 1.5 MP: it stores the 128-channel full-resolution map and its copies (about 2 GB per megapixel)
  and wasm32 has 4 GB. A lazy pyramid on the CPU backend, as on the GPU backends, would lift that.
- Still open: no profiler on WebGPU (timestamp queries would give GPU time per dispatch), bind groups are
  created per dispatch, and the browser numbers were taken with the pane hidden. Not checked: Safari and
  Firefox (JSPI), Dawn on Vulkan or Direct3D 12 (the laptop).

## Notes from M7

- The site follows just-ba's: Sphinx with MyST, its own theme on Sphinx's basic theme (`docs/_theme/jkp`, just-ba's
  layout and stylesheet with another accent), and `.github/workflows/docs.yml`, started by hand, which builds the
  WebAssembly module (Emscripten 7.0.0 from emsdk, emdawnwebgpu from `tools/fetch_webgpu.sh browser`), builds the
  site with warnings as errors and, with `deploy=true`, uploads it with wrangler. Same secrets and variable
  names as just-ba (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_PAGES_PROJECT`, default
  `just-keypoints`).
- Public pages carry no references to this plan, milestones, commit hashes or the repository (it is private),
  as just-ba's rule for its site.
- The demo and benchmark pages are static files in `docs/_extra/` (copied as they are), sharing the theme's
  stylesheet. `conf.py` copies the module, `jkp.js` and the weights into `lib/` after the build, from
  `JKP_WASM_DIR` (default `build/wasm/web`); without them the build warns, so `-W` fails.
- Stage timings: the plan asked for ms per stage (upload, network, top-K, descriptors). The library reports no
  stage times, and timing stages from outside would need a GPU round trip per stage. The page shows its own steps
  (prepare, extract, match) and the first run (shader compilation). Per-stage GPU time needs WebGPU timestamp
  queries in the backend (the M6 open point).
- Warm timings: the first run of a new extractor compiles the shaders (45-60 ms with the browser's shader cache
  warm, about 600 ms cold), and the first run at a new image size allocates its buffers (66 ms against 16 ms
  warm for 1241x376). The demo runs both untimed; the benchmark page has warm-up runs.
- The matcher (mutual nearest neighbours, Lowe's ratio test both ways) is plain JavaScript in a worker:
  about 70 ms for 1000 x 1000 keypoints. A GPU matcher would be faster but stays out of scope.
- The browser pane, when hidden, delays an `<img>`'s decoding and `OffscreenCanvas.convertToBlob` by seconds:
  the pages load images with `fetch` + `createImageBitmap` and draw thumbnails on canvases.
- Sample images: two TUM RGB-D frames (CC BY 4.0) and two KITTI frames (CC BY-NC-SA 3.0, non-commercial) from
  the ALIKED repository, as JPEG; credited on the references page and in the demo's footer. The St Paul's and
  San Marco photos there were left out: their source and licence are not stated.
- Fallback without WebGPU: checked on the live demo with `navigator.gpu` hidden before the page's scripts run
  (the page in a same-origin frame; a permissions policy of `webgpu 'none'` on the frame did not hide the
  adapter in Chromium 156). The page shows its notice, picks the CPU backend, turns FP16 off (no shader-f16)
  and reports "no WebGPU adapter"; TUM pair, ALIKED top-k 1000: the same 229 matches, 1.2-1.8 s per image at
  640x480.
- Not checked: Edge, Safari, Firefox, Chrome on Windows and Linux, a browser without WebGPU at all (only the
  hidden `navigator.gpu` above), and the camera (needs a device and a permission prompt).
- RaCo in the demo: added with R6; see the notes from R6.

## Open questions (decide during implementation)

- Tensor layout for speed: NCHW vs NHWC on CUDA. (WGSL: storage buffers, one per tensor, settled in M6.)
- WebGPU limits: settled in M6 for the sizes tried (up to 2075x1491): the device asks for the adapter's
  buffer limits and no kernel binds more than seven buffers. Past 4 GB per tensor it would need tiling.
- Whether to add `aliked-t16` as a second, faster web-friendly variant (same op set, different widths).
- Descriptor output quantization for the downstream matcher.
- RaCo in FP16 (see the notes from R5).
