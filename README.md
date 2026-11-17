# just-keypoints

Keypoint and descriptor extraction (ALIKED) with minimal dependencies, for feature matching and bundle adjustment.
No PyTorch, no ONNX Runtime. Plan and status: [PLAN.md](PLAN.md).

> [!IMPORTANT]
> **Use [ALIKED](https://github.com/Shiaoming/ALIKED) and [RaCo](https://github.com/cvg/RaCo).** This repository
> is a test bed for dependency-free inference and GPU kernels, not a replacement for them. It adds nothing to
> either except speed and fewer dependencies: no new networks, training or features, and only a subset of what
> they offer (one ALIKED variant, `aliked-n16`; RaCo is planned). The methods, the weights, and the credit for
> them are their authors'. For real work, ALIKED and RaCo are the projects to depend on.

Status: the CPU reference backend and the CUDA backend work and match the original ALIKED. The CUDA backend is
not optimized yet. Metal and WebGPU backends are planned.

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else for the CPU backend.
The CUDA backend is built when the CUDA toolkit (`nvcc`) is found; it uses no other library.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

| Option | Default | Meaning |
|---|---|---|
| `JKP_EMBED_WEIGHTS` | `ON` | compile the weights into the library (adds ~2.7 MB); `OFF` = give a path or memory blob at run time |
| `JKP_CUDA` | `AUTO` | CUDA backend: `AUTO` = build it when a CUDA compiler is found, `ON` = require it, `OFF` |
| `JKP_BUILD_CLI` | `ON` | build the `jkp` tool |
| `JKP_BUILD_TESTS` | `ON` | build tests |

## Command line

```sh
build/release/jkp extract image.png --out features.jkpt    # keypoints, scores, descriptors
build/release/jkp bench image.png --runs 10
build/release/jkp bench image.png --backend cuda
```

Options: `--backend cpu|cuda`, `--device N`, `--top-k K`, `--scores-th T`, `--n-limit N`, `--weights file.jkpw`, `--threads N`, `--gray`.

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
python tools/dump_golden.py third_party/ALIKED/assets/tum/*.png   # then: ctest --test-dir build/release
```

`ctest` runs the golden files against every backend in the build, and compares the CUDA backend with the
CPU backend on generated images (no golden files or Python needed for that one):

```sh
ctest --test-dir build/release -R cuda_vs_cpu
```

## Licenses

Code in this repository: to be decided. Weights in `weights/` come from ALIKED (BSD-3-Clause, see
`weights/LICENSE-ALIKED`). `cli/stb_image.h` is public domain / MIT and is used only by the CLI.
