# Quick start

## From C or C++

```c
#include "jkp/jkp.h"

jkp_config config;
jkp_config_default(&config);              /* CPU, aliked-n16, embedded weights */
if (jkp_backend_available(JKP_BACKEND_METAL)) config.backend = JKP_BACKEND_METAL;
config.top_k = 1000;                      /* the 1000 best keypoints; 0 = every keypoint above a threshold */
jkp_ctx* ctx = jkp_create(&config);
if (!ctx) { fprintf(stderr, "%s\n", jkp_last_error()); return 1; }

jkp_image image = {pixels, width, height, 0, JKP_FORMAT_RGB8};
jkp_result r;
if (jkp_extract(ctx, &image, &r) == 0) {
  for (int i = 0; i < r.n; ++i) {
    float x = r.xy[2 * i], y = r.xy[2 * i + 1];   /* pixels, (0, 0) = centre of the top-left pixel */
    float s = r.score[i];
    const float* d = r.desc + i * r.dim;          /* 128 floats, L2-normalized */
  }
}
jkp_destroy(ctx);
```

Link `libjkp.a` (and, for the GPU backends, the CUDA runtime, Metal.framework or Dawn). Create a context once
and reuse it: it holds the weights and the GPU buffers, so later images run without allocating. The result's
arrays belong to the context and stay valid until the next `jkp_extract`.

Pixels can be `GRAY8`, `RGB8` or `RGBA8`, with any row stride. The image can be any size: it is padded on the
device to a multiple of 32.

## Choosing keypoints

| Setting | Meaning |
|---|---|
| `top_k > 0` | keep the `top_k` keypoints with the best scores |
| `top_k <= 0` | keep every keypoint with a score above `score_threshold` (default 0.2), at most `max_keypoints` (default 5000), the best ones first when there are more |

Both are the original ALIKED's selection modes, and give the original's keypoints.

## FP32 and FP16

FP32 is the default on every backend and matches the original network. On the GPU backends
`config.precision = JKP_PRECISION_FP16` stores feature maps and weights as 16-bit floats: 10-20 % faster, and
slightly different results, with about 99.8 % of the keypoints the same. WebGPU needs the adapter to have the
`shader-f16` feature. See [Accuracy and speed](accuracy-and-speed.md).

## From the command line

```bash
jkp extract image.png --out features.jkpt                 # keypoints, scores, descriptors
jkp extract image.png --backend metal --top-k 1000
jkp bench image.png --backend cuda --precision fp16 --runs 20
```

| Option | Meaning |
|---|---|
| `--backend cpu\|cuda\|metal\|webgpu` | backend (default `cpu`) |
| `--device N` | GPU index |
| `--precision fp32\|fp16` | precision (default `fp32`) |
| `--top-k K` | keep the K best keypoints |
| `--scores-th T`, `--n-limit N` | threshold mode: score threshold and the cap |
| `--weights file.jkpw` | weights file instead of the embedded weights |
| `--threads N` | CPU worker threads (default: all) |
| `--gray` | read the image as grayscale |

The output file uses the same container as the weights ([Weights format](weights-format.md)), with the
tensors `xy`, `score` and `desc`.

## RaCo

RaCo is a second detector: it returns, per keypoint, a ranker score and a 2x2 covariance in pixels² (usable as a
weight in bundle adjustment), and no descriptors. It runs on the CPU and Metal backends (FP32), and gives the
original RaCo's results.

```bash
jkp extract image.png --model raco --backend metal
jkp extract image.png --model raco --backend metal --descriptor aliked-n16   # with ALIKED descriptors
```

A build that embeds only ALIKED (`JKP_EMBED_MODELS=aliked-n16`) needs `--weights raco.jkpw` as well.

In the C API: `config.model = "raco"`, and `config.descriptor = "aliked-n16"` to describe its keypoints with
ALIKED. `skip_ranker` and `skip_covariance` leave out a head; the covariance head is about three quarters of the
run time.
