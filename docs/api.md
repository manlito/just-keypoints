# C API

The whole public interface is one header, `jkp/jkp.h`, in C. Errors are reported by return values; the message
for the last failure on the calling thread comes from `jkp_last_error()`.

## Functions

| Function | Meaning |
|---|---|
| `void jkp_config_default(jkp_config*)` | fills a configuration with the defaults: CPU, `aliked-n16`, threshold 0.2, at most 5000 keypoints, FP32 |
| `int jkp_backend_available(jkp_backend)` | 1 when the backend is compiled into this build (whether the device works is checked by `jkp_create`) |
| `jkp_ctx* jkp_create(const jkp_config*)` | loads the weights and prepares the backend; `NULL` on failure |
| `int jkp_extract(jkp_ctx*, const jkp_image*, jkp_result*)` | runs one image; 0 on success |
| `void jkp_destroy(jkp_ctx*)` | frees the context and its results |
| `const char* jkp_last_error(void)` | message for the last failure on the calling thread |

A context must not be used from several threads at once. Use one context per thread, or per GPU stream.

## `jkp_config`

| Field | Meaning |
|---|---|
| `backend` | `JKP_BACKEND_CPU`, `JKP_BACKEND_CUDA`, `JKP_BACKEND_METAL` or `JKP_BACKEND_WEBGPU` |
| `model` | `"aliked-n16"` (default) or `"raco"`; selects the embedded weights when none are given |
| `weights_data`, `weights_size` | weights file in memory (copied by `jkp_create`) |
| `weights_path` | weights file on disk; used when no memory blob is given. Neither set: the embedded weights |
| `top_k` | `> 0`: keep the `top_k` best keypoints. Otherwise threshold mode |
| `score_threshold` | threshold mode: keep keypoints scoring above it (default 0.2) |
| `max_keypoints` | threshold mode: at most this many, best first when there are more (default 5000) |
| `num_threads` | CPU backend worker threads; 0 = all hardware threads |
| `device` | CUDA and Metal: device index, 0 = the first |
| `precision` | `JKP_PRECISION_FP32` (default, matches the original) or `JKP_PRECISION_FP16` (GPU backends); `jkp_create` fails when the backend does not support it |
| `descriptor` | a second model that describes the keypoints of a detector without descriptors: `model = "raco"`, `descriptor = "aliked-n16"` |
| `descriptor_weights_data`, `descriptor_weights_size`, `descriptor_weights_path` | its weights, as above |
| `skip_ranker`, `skip_covariance` | RaCo: non-zero leaves out that head; its output is then `NULL` |
| `webgpu_instance`, `webgpu_device` | WebGPU: a `WGPUInstance` and a `WGPUDevice` from it, to run on the caller's device. Both `NULL`: the backend creates its own. The device must outlive the context |

RaCo always keeps the `top_k` best keypoints (2048 when `top_k <= 0`); it has no score threshold.

## `jkp_image`

| Field | Meaning |
|---|---|
| `data` | pixels, top-left origin, in host memory |
| `width`, `height` | in pixels; any size |
| `stride` | bytes per row; 0 = `width` times the bytes per pixel |
| `format` | `JKP_FORMAT_GRAY8`, `JKP_FORMAT_RGB8` or `JKP_FORMAT_RGBA8` |

## `jkp_result`

The arrays belong to the context and stay valid until the next `jkp_extract` or `jkp_destroy` on it.

| Field | Meaning |
|---|---|
| `n` | number of keypoints |
| `dim` | descriptor size: 128 for `aliked-n16`, 0 for RaCo without a descriptor model |
| `xy` | `[n][2]` pixel coordinates, x right and y down, (0, 0) = the centre of the top-left pixel; subpixel |
| `score` | `[n]` detection score; ALIKED: in [0, 1]; RaCo: a probability, summing to 1 over the image |
| `desc` | `[n][dim]` descriptors, L2-normalized; `NULL` when `dim` is 0 |
| `rank` | `[n]` RaCo ranker score, higher is better; `NULL` for other models |
| `cov` | `[n][2][2]` RaCo keypoint covariance in pixels²; `NULL` for other models |

## JavaScript

The browser wrapper `jkp.js` has the same model: `create(options)` returns an extractor (a context), whose
`extract(image)` returns `{n, dim, xy, score, desc}` as `Float32Array` copies, and `destroy()` frees it. See
[In the browser](browser.md).
