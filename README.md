# just-keypoints

Keypoint and descriptor extraction (ALIKED) with minimal dependencies, for feature matching and bundle adjustment.
No PyTorch, no ONNX Runtime. Plan and status: [PLAN.md](PLAN.md).

> [!IMPORTANT]
> **Use [ALIKED](https://github.com/Shiaoming/ALIKED) and [RaCo](https://github.com/cvg/RaCo).** This repository
> is a test bed for dependency-free inference and GPU kernels, not a replacement for them. It adds nothing to
> either except speed and fewer dependencies: no new networks, training or features, and only a subset of what
> they offer (inference only, and one ALIKED variant, `aliked-n16`). The methods, the weights, and the credit
> for them are their authors'. For real work, ALIKED and RaCo are the projects to depend on.

Documentation: <https://just-keypoints.pages.dev> ([live demo](https://just-keypoints.pages.dev/demo/),
[benchmark](https://just-keypoints.pages.dev/bench/)).

Status: the CPU reference backend and the CUDA, Metal and WebGPU backends work and match the original ALIKED.
WebGPU runs natively (Dawn) and in the browser (WebAssembly).

Milliseconds per image with 1000 keypoints (image in host memory to keypoints and descriptors in host memory):

| Size | Metal FP32 | Metal FP16 | PyTorch MPS | CUDA FP32 | CUDA FP16 | PyTorch CUDA |
|---|---|---|---|---|---|---|
| 640x480 | 8.3 | 7.2 | 13.1 | 15.2 | 13.4 | 15.6 |
| 1024x768 | 13.8 | 11.3 | 28.9 | 26.8 | 24.5 | 31.8 |
| 1920x1080 | 30.2 | 23.3 | 74.8 | 59.1 | 53.9 | 77.3 |

Metal and MPS on an M5 Pro, CUDA on an RTX 2070 Max-Q laptop; PyTorch runs the original ALIKED. Details in [PLAN.md](PLAN.md), M5.
WebGPU on the same M5 Pro (native, Dawn on Metal): 10.1 / 17.1 / 38.0 ms in FP32, 8.8 / 15.2 / 33.3 ms in FP16;
in Chrome about 12 ms at 640x480 (PLAN.md, M6).

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else for the CPU backend.
The CUDA backend is built when the CUDA toolkit (`nvcc`) is found; it uses no other library.
The Metal backend is built on macOS; it uses only Metal.framework and compiles its shaders at run time, so
no Metal toolchain is needed.
The WebGPU backend is built when Dawn's prebuilt release is found (see [WebGPU](#webgpu-and-the-browser)).

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

| Option | Default | Meaning |
|---|---|---|
| `JKP_EMBED_WEIGHTS` | `ON` | compile the weights into the library (adds ~4.7 MB for both models); `OFF` = give a path or memory blob at run time |
| `JKP_EMBED_MODELS` | `aliked-n16;raco` | models to embed (names in `weights/`); `aliked-n16` alone adds ~2.7 MB |
| `JKP_CUDA` | `AUTO` | CUDA backend: `AUTO` = build it when a CUDA compiler is found, `ON` = require it, `OFF` |
| `JKP_METAL` | `AUTO` | Metal backend: `AUTO` = build it on Apple platforms, `ON` = require it, `OFF` |
| `JKP_WEBGPU` | `AUTO` | WebGPU backend: `AUTO` = build it when Dawn (or, under Emscripten, emdawnwebgpu) is found, `ON` = require it, `OFF` |
| `JKP_BUILD_WASM` | `ON` | under Emscripten: build the WebAssembly module, `web/jkp.js` and the test page |
| `JKP_BUILD_CLI` | `ON` | build the `jkp` tool |
| `JKP_BUILD_TESTS` | `ON` | build tests |
| `JKP_BUILD_PYTHON` | `OFF` | build the Python package `just_keypoints` (needs nanobind; see [Python](#python)) |

## Python

```sh
pip install just-keypoints          # CPU; Metal on macOS (arm64, 13 or later)
pip install just-keypoints-cu12     # or: CPU and CUDA 12 (Linux, Windows x86_64; Pascal and later)
pip install just-keypoints-cu13     # or: CPU and CUDA 13 (Turing and later)
```

```python
import just_keypoints as jkp
extractor = jkp.Extractor(backend="cuda" if jkp.backend_available("cuda") else "cpu", top_k=2000)
f = extractor.extract(image)        # uint8 (h, w), (h, w, 3) or (h, w, 4)
f.xy, f.score, f.desc                # NumPy float32 copies: (n, 2), (n,), (n, 128)
```

The keyword arguments of `Extractor` are the fields of `jkp_config` (`model="raco"`, `descriptor="aliked-n16"`,
`precision="fp16"`, ...). Both models' weights are built in, and the CUDA packages need only an NVIDIA driver.
From a checkout: `pip install .`, or with CMake `-DJKP_BUILD_PYTHON=ON -DPython_EXECUTABLE=<python with
nanobind>` and `ctest -R python`. Every merge to main is published to PyPI with the next patch version
([PLAN.md](PLAN.md), phase 7).

## Command line

```sh
build/release/jkp extract image.png --out features.jkpt    # keypoints, scores, descriptors
build/release/jkp bench image.png --runs 10
build/release/jkp bench image.png --backend cuda
build/release/jkp bench image.png --backend metal
build/release/jkp bench image.png --backend metal --precision fp16
build/release/jkp bench image.png --backend webgpu
JKP_METAL_PROFILE=1 build/release/jkp bench image.png --backend metal   # GPU time per operation
JKP_CUDA_PROFILE=1 build/release/jkp bench image.png --backend cuda     # kernel time per operation
```

Options: `--backend cpu|cuda|metal|webgpu`, `--device N`, `--precision fp32|fp16`, `--top-k K`, `--scores-th T`, `--n-limit N`, `--weights file.jkpw`, `--threads N`, `--gray`.

## C API

```c
#include "jkp/jkp.h"

jkp_config config;
jkp_config_default(&config);              /* CPU, aliked-n16, embedded weights */
if (jkp_backend_available(JKP_BACKEND_CUDA)) config.backend = JKP_BACKEND_CUDA;
jkp_ctx* ctx = jkp_create(&config);

jkp_image image = {pixels, width, height, 0, JKP_FORMAT_RGB8};
jkp_result r;
if (jkp_extract(ctx, &image, &r) == 0) {
  /* r.n keypoints: r.xy[2*i], r.xy[2*i+1], r.score[i], r.desc[i*r.dim ...] */
}
jkp_destroy(ctx);
```

## Precision

FP32 is the default on every backend and matches the original network: the same keypoints, positions within
3e-4 px. `--precision fp16` (C API: `config.precision = JKP_PRECISION_FP16`, GPU backends only; WebGPU needs the
`shader-f16` feature) stores feature
maps and weights as 16-bit floats. It is 10-20 % faster and its results differ slightly: about 99.8 % of the
keypoints are the same, those move by about 0.001 px (median), and their descriptors differ by about 1e-5 in
cosine similarity. `ctest -R fp16` measures this on your machine.

## Validation against the original

See [tools/README.md](tools/README.md). In short:

```sh
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract"
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract --backend cuda"
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract --backend metal"
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract --backend webgpu"
python tools/dump_golden.py third_party/ALIKED/assets/tum/*.png   # then: ctest --test-dir build/release
```

`ctest` runs the golden files against every backend in the build, and compares each GPU backend with the
CPU backend on generated images, operation by operation (`_ops`) and for the whole network (`_vs_cpu`); these
need no golden files or Python:

```sh
ctest --test-dir build/release -R _vs_cpu
```

## WebGPU and the browser

The WebGPU backend is one C++ source over `webgpu.h`: natively it links Dawn, in the browser Emscripten's
emdawnwebgpu. Both come from the same Dawn release (v20261113.214538, the one just-ba uses), unpacked under
`third_party/webgpu` (not committed):

```text
third_party/webgpu/dawn/Dawn-7daf3b0bee2c73621001ab15b256b5a84fb0b45f-macos-latest-Release/   (lib/cmake/Dawn)
third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg/                                             (emdawnwebgpu.port.py)
```

`tools/fetch_webgpu.sh` downloads both (`browser` or `native` for one). Or point `JKP_DAWN_DIR` /
`JKP_EMDAWNWEBGPU_DIR` elsewhere. The browser build needs Emscripten 7.0.0 (Homebrew)
and Python 3.10 or later for it (`EMSDK_PYTHON` when the default `python3` is older):

```sh
emcmake cmake -S . -B build/wasm -G Ninja -DCMAKE_BUILD_TYPE=Release -DJKP_EMBED_WEIGHTS=OFF -DJKP_BUILD_CLI=OFF -DJKP_BUILD_TESTS=OFF
cmake --build build/wasm                       # build/wasm/web: jkp_wasm.js/.wasm, jkp.js, test.html, weights
python tools/web_testdata.py --jkp build/release/jkp --out build/wasm/web/testdata third_party/ALIKED/assets/tum/*.png
python3 -m http.server 8791 --bind 127.0.0.1 --directory build/wasm/web   # then open /test.html in Chrome
```

```js
import * as jkp from './jkp.js';
const extractor = await jkp.create({ weightsUrl: 'aliked-n16.jkpw', topK: 1000 });  // backend 'webgpu' or 'cpu'
const { n, dim, xy, score, desc } = await extractor.extract(imageBitmap);         // Float32Arrays
```

The module is single-threaded and waits for the GPU through JSPI (Chrome and Edge). An app that also runs
other GPU work passes its own device: `config.webgpu_instance` and `config.webgpu_device`. An app that links
just-keypoints with other libraries into one WebAssembly module links the static library (`libjkp.a`) and the C
API, with the same settings (JSPI, `-fwasm-exceptions`).

## Documentation site and live demo

`docs/` holds the documentation site (Sphinx) with a live demo and a benchmark page that run the browser build
on WebGPU, published at <https://just-keypoints.pages.dev>. Build and deploy: [docs/README.md](docs/README.md).

## RaCo detector

[RaCo](https://github.com/cvg/RaCo) is a second model: a detector that returns, per keypoint, a ranker score
and a 2x2 covariance in pixels², and no descriptors. It runs on the CPU, CUDA, Metal and WebGPU backends (WebGPU
natively and in the browser) and matches the original on each.

```sh
build/release/jkp extract image.png --model raco --out features.jkpt
build/release/jkp extract image.png --model raco --backend cuda --out features.jkpt
build/release/jkp extract image.png --model raco --backend metal --out features.jkpt
build/release/jkp extract image.png --model raco --backend webgpu --out features.jkpt
python tools/compare.py --model raco image.png --b "cmd:build/release/jkp extract --model raco --backend metal"
```

Add `--descriptor aliked-n16` (C API: `config.descriptor`) to get ALIKED descriptors for the RaCo keypoints;
this runs both networks. `--no-covariance` and `--no-ranker` (`skip_covariance`, `skip_ranker`) leave out a head:
the covariance head is about three quarters of the run time on the CPU and more than half on the GPUs.

The default build embeds both models; with `-DJKP_EMBED_MODELS=aliked-n16` (RaCo adds ~2 MB) pass
`--weights weights/raco.jkpw` instead. RaCo runs in FP32 only.
In the C API set `config.model = "raco"`; the result then has `rank` and `cov`, and `desc` is `NULL` unless a descriptor model is set.
In the browser: `jkp.create({ weightsUrl: 'raco.jkpw', descriptorWeightsUrl: 'aliked-n16.jkpw' })`; results add `rank` and `cov`.
`top_k` is the number of keypoints (default 2048); there is no score threshold.

## Licenses

Code in this repository: Apache-2.0 (`LICENSE`). Weights in `weights/` come from ALIKED (BSD-3-Clause, see
`weights/LICENSE-ALIKED`) and RaCo (Apache-2.0, see `weights/LICENSE-RACO`). `cli/stb_image.h` is public domain / MIT and is used only by the CLI.
