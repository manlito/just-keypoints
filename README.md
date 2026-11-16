# just-keypoints

Keypoint and descriptor extraction (ALIKED) with minimal dependencies, for feature matching and bundle adjustment.
No PyTorch, no ONNX Runtime. Plan and status: [PLAN.md](PLAN.md).

Status: the CPU reference backend, the CUDA backend and the Metal backend work and match the original ALIKED.
The GPU backends are not optimized yet. A WebGPU backend is planned.

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else for the CPU backend.
The CUDA backend is built when the CUDA toolkit (`nvcc`) is found; it uses no other library.
The Metal backend is built on macOS; it uses only Metal.framework and compiles its shaders at run time, so
no Metal toolchain is needed.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

| Option | Default | Meaning |
|---|---|---|
| `JKP_EMBED_WEIGHTS` | `ON` | compile the weights into the library (adds ~2.7 MB); `OFF` = give a path or memory blob at run time |
| `JKP_CUDA` | `AUTO` | CUDA backend: `AUTO` = build it when a CUDA compiler is found, `ON` = require it, `OFF` |
| `JKP_METAL` | `AUTO` | Metal backend: `AUTO` = build it on Apple platforms, `ON` = require it, `OFF` |
| `JKP_BUILD_CLI` | `ON` | build the `jkp` tool |
| `JKP_BUILD_TESTS` | `ON` | build tests |

## Command line

```sh
build/release/jkp extract image.png --out features.jkpt    # keypoints, scores, descriptors
build/release/jkp bench image.png --runs 10
build/release/jkp bench image.png --backend cuda
build/release/jkp bench image.png --backend metal
JKP_METAL_PROFILE=1 build/release/jkp bench image.png --backend metal   # GPU time per operation
```

Options: `--backend cpu|cuda|metal`, `--device N`, `--top-k K`, `--scores-th T`, `--n-limit N`, `--weights file.jkpw`, `--threads N`, `--gray`.

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

## Validation against the original

See [tools/README.md](tools/README.md). In short:

```sh
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract"
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract --backend cuda"
python tools/compare.py third_party/ALIKED/assets/tum/*.png --b "cmd:build/release/jkp extract --backend metal"
python tools/dump_golden.py third_party/ALIKED/assets/tum/*.png   # then: ctest --test-dir build/release
```

`ctest` runs the golden files against every backend in the build, and compares each GPU backend with the
CPU backend on generated images (no golden files or Python needed for that one):

```sh
ctest --test-dir build/release -R _vs_cpu
```

## RaCo detector

[RaCo](https://github.com/cvg/RaCo) is a second model: a detector that returns, per keypoint, a ranker score
and a 2x2 covariance in pixels², and no descriptors. It runs on the CPU backend for now and matches the original.

```sh
build/release/jkp extract image.png --model raco --weights weights/raco.jkpw --out features.jkpt
python tools/compare.py --model raco image.png --b "cmd:build/release/jkp extract --weights weights/raco.jkpw"
```

Add `--descriptor aliked-n16` (C API: `config.descriptor`) to get ALIKED descriptors for the RaCo keypoints;
this runs both networks. `--no-covariance` and `--no-ranker` (`skip_covariance`, `skip_ranker`) leave out a head:
the covariance head is about three quarters of the run time.

`--weights` is not needed when the build embeds the model: `-DJKP_EMBED_MODELS="aliked-n16;raco"` (adds ~2 MB).
In the C API set `config.model = "raco"`; the result then has `rank` and `cov`, and `desc` is `NULL` unless a descriptor model is set.
`top_k` is the number of keypoints (default 2048); there is no score threshold.

## Licenses

Code in this repository: to be decided. Weights in `weights/` come from ALIKED (BSD-3-Clause, see
`weights/LICENSE-ALIKED`) and RaCo (Apache-2.0, see `weights/LICENSE-RACO`). `cli/stb_image.h` is public domain / MIT and is used only by the CLI.
