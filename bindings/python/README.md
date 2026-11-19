# just-keypoints

Keypoints and descriptors from [ALIKED](https://github.com/Shiaoming/ALIKED) (`aliked-n16`) and keypoints with
ranker scores and covariances from [RaCo](https://github.com/cvg/RaCo), with minimal dependencies: hand-written
CPU, Metal and CUDA kernels, no PyTorch, no ONNX Runtime. In FP32 the results match the original networks.

This is a test bed for dependency-free inference, not a replacement for ALIKED or RaCo: the methods, the
weights and the credit are their authors'. For real work, depend on them.

## Install

| Package | Platforms | Backends |
|---|---|---|
| `pip install just-keypoints` | Linux x86_64, macOS arm64 (13 or later), Windows x86_64 | CPU; Metal on macOS |
| `pip install just-keypoints-cu12` | Linux x86_64, Windows x86_64 | CPU and CUDA 12 |
| `pip install just-keypoints-cu13` | Linux x86_64, Windows x86_64 | CPU and CUDA 13 |

Install one of them: all three provide the `just_keypoints` module. The CUDA packages link the CUDA runtime
statically and need only an NVIDIA driver. CUDA 12 runs on Pascal (sm_60) and later, CUDA 13 on Turing (sm_75)
and later. Python 3.12 or later. Both models' weights are built in.

```python
import just_keypoints as jkp
print(jkp.__version__, jkp.compiled_backends())
```

## Use

```python
import numpy as np
import just_keypoints as jkp

backend = "cuda" if jkp.backend_available("cuda") else "metal" if jkp.backend_available("metal") else "cpu"
extractor = jkp.Extractor(backend=backend, top_k=2000)   # aliked-n16, built-in weights
f = extractor.extract(image)            # uint8 (h, w), (h, w, 3) RGB or (h, w, 4) RGBA
f.xy, f.score, f.desc                    # (n, 2), (n,), (n, 128) float32 copies

raco = jkp.Extractor(backend=backend, model="raco", top_k=2048, descriptor="aliked-n16")
r = raco.extract(image)
r.xy, r.rank, r.cov, r.desc              # (n, 2), (n,), (n, 2, 2) in pixels², (n, 128)
```

Pixel coordinates are x right, y down, with (0, 0) the centre of the top-left pixel. `precision="fp16"` is
faster on the GPU backends (ALIKED only) with slightly different results. Without `top_k`, ALIKED keeps every
keypoint with `score > score_threshold` (0.2), at most `max_keypoints` (5000). An `Extractor` is made once and
used for many images; calls from several threads are serialized. Errors (no such device, bad weights) raise
`RuntimeError`.

Documentation: <https://just-keypoints.pages.dev>.

Licences: the built-in weights are ALIKED's (BSD-3-Clause) and RaCo's (Apache-2.0); both licence files are in
the package.
