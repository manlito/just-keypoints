# just-keypoints

Keypoints and descriptors from the ALIKED network, with minimal dependencies, for feature matching and
bundle adjustment. No PyTorch, no ONNX Runtime, no cuDNN: one C++ library with hand-written kernels for
each backend.

- **Backends**: the CPU (plain C++, the reference), CUDA on NVIDIA GPUs, Metal on Apple Silicon, and
  WebGPU, natively and in the browser through WebAssembly.
- **Same results as the original**: in FP32 every backend finds the same keypoints as the original
  PyTorch ALIKED, with positions within 3e-4 px and descriptors within 1e-8 in cosine similarity.
  FP16 is an opt-in fast mode on the GPU backends.
- **Fast**: 8 ms per 640x480 image on an M5 Pro (Metal), 15 ms on an RTX 2070 laptop (CUDA), about 12 ms
  in Chrome (WebGPU), each with 1000 keypoints and descriptors.
- **One small C API**: give it pixels, get back keypoint positions, scores and 128-dimensional,
  L2-normalized descriptors.
- **Not in scope**: matching. The library stops at keypoints and descriptors; the demo's matcher lives in
  the demo.

:::{admonition} Try it in your browser
<a href="demo/index.html">The live demo</a> runs the library on your GPU through WebGPU: pick two images, see the
keypoints, the matches between them and the time each step takes. <a href="bench/index.html">The benchmark page</a>
times a fixed set of images, so numbers compare across machines. Both need a browser with WebGPU and
JavaScript promise integration (Chrome or Edge 137 or later).
:::

```{toctree}
:maxdepth: 2
:caption: Guide

install
quickstart
browser
accuracy-and-speed
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
weights-format
references
```
