# just-keypoints

Keypoint and descriptor extraction (ALIKED) with minimal dependencies, for feature matching and bundle adjustment.
No PyTorch, no ONNX Runtime. Plan and status: [PLAN.md](PLAN.md).

Status: the CPU reference backend works and matches the original ALIKED. CUDA, Metal and WebGPU backends are planned.

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

| Option | Default | Meaning |
|---|---|---|
| `JKP_EMBED_WEIGHTS` | `ON` | compile the weights into the library (adds ~2.7 MB); `OFF` = give a path or memory blob at run time |
| `JKP_BUILD_CLI` | `ON` | build the `jkp` tool |
| `JKP_BUILD_TESTS` | `ON` | build tests |

## Command line

```sh
build/release/jkp extract image.png --out features.jkpt    # keypoints, scores, descriptors
build/release/jkp bench image.png --runs 10
```

Options: `--top-k K`, `--scores-th T`, `--n-limit N`, `--weights file.jkpw`, `--threads N`, `--gray`.

## C API

```c
#include "jkp/jkp.h"

jkp_config config;
jkp_config_default(&config);              /* CPU, aliked-n16, embedded weights */
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
python tools/dump_golden.py third_party/ALIKED/assets/tum/*.png   # then: ctest --test-dir build/release
```

## Licenses

Code in this repository: to be decided. Weights in `weights/` come from ALIKED (BSD-3-Clause, see
`weights/LICENSE-ALIKED`). `cli/stb_image.h` is public domain / MIT and is used only by the CLI.
