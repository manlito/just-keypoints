# Install

## Python

```bash
pip install just-keypoints          # CPU; Metal on macOS (arm64, 13 or later)
pip install just-keypoints-cu12     # or: CPU and CUDA 12 (Linux, Windows x86_64)
pip install just-keypoints-cu13     # or: CPU and CUDA 13 (Linux, Windows x86_64)
```

Install one of them: all three provide the `just_keypoints` module, for Python 3.12 and later. Both models'
weights are built in. The CUDA packages link the CUDA runtime statically and need only an NVIDIA driver;
CUDA 12 runs on Pascal GPUs and later, CUDA 13 on Turing and later.

```python
import just_keypoints as jkp

extractor = jkp.Extractor(backend="cuda" if jkp.backend_available("cuda") else "cpu", top_k=2000)
features = extractor.extract(image)                  # uint8, (h, w), (h, w, 3) RGB or (h, w, 4) RGBA
features.xy, features.score, features.desc           # (n, 2), (n,), (n, 128) float32
```

The keyword arguments of `Extractor` are the fields of the C API's `jkp_config`: `model="raco"` with
`descriptor="aliked-n16"`, `precision="fp16"`, `score_threshold`, `max_keypoints`, `device`, and `weights`
(a `.jkpw` path or its bytes). RaCo's results add `rank` and `cov`.

## From source

just-keypoints is built from source with CMake 3.16 or later and a C++17 compiler. The CPU backend needs
nothing else. Each GPU backend is built when its toolkit is found, and uses no library beyond it.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

This gives the static library `libjkp.a`, the single public header `jkp/jkp.h` and the `jkp` command-line
tool. By default the ALIKED weights (2.7 MB) are compiled into the library, so nothing has to be shipped
next to it.

## Backends

| Backend | Built when | Notes |
|---|---|---|
| CPU | always | Plain C++, multi-threaded. The reference the others are checked against. |
| CUDA | the CUDA toolkit (`nvcc`) is found | Needs only the CUDA runtime library. |
| Metal | on macOS | Needs only Metal.framework; the shaders are compiled at run time, so no Metal toolchain is needed. |
| WebGPU | Dawn's prebuilt release is found | The same source runs natively (Dawn on Metal, Vulkan or Direct3D 12) and in the browser. |

## Options

| Option | Default | Meaning |
|---|---|---|
| `JKP_EMBED_WEIGHTS` | `ON` | compile the weights into the library; `OFF` = give a path or a memory blob at run time |
| `JKP_EMBED_MODELS` | `aliked-n16;raco` | models to embed; `aliked-n16` alone leaves out RaCo's 2 MB |
| `JKP_CUDA` | `AUTO` | `AUTO` = build it when a CUDA compiler is found, `ON` = require it, `OFF` |
| `JKP_METAL` | `AUTO` | `AUTO` = build it on Apple platforms, `ON` = require it, `OFF` |
| `JKP_WEBGPU` | `AUTO` | `AUTO` = build it when Dawn (or, under Emscripten, emdawnwebgpu) is found, `ON` = require it, `OFF` |
| `JKP_BUILD_WASM` | `ON` | under Emscripten: build the WebAssembly module and its JavaScript wrapper |
| `JKP_BUILD_CLI` | `ON` | build the `jkp` tool |
| `JKP_BUILD_TESTS` | `ON` | build the tests |

## WebGPU

Natively the WebGPU backend links Dawn; in the browser it uses Emscripten's emdawnwebgpu. Both come from the
same Dawn release, v20261113.214538. The repository has a script that downloads them into the build tree:

```bash
tools/fetch_webgpu.sh            # emdawnwebgpu, and Dawn for this machine
```

Point `JKP_DAWN_DIR` or `JKP_EMDAWNWEBGPU_DIR` elsewhere to use another copy.

## The browser build

The browser build needs Emscripten 7.0.0, and Python 3.10 or later for Emscripten (set `EMSDK_PYTHON` when the
default `python3` is older):

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release -DJKP_EMBED_WEIGHTS=OFF -DJKP_BUILD_CLI=OFF -DJKP_BUILD_TESTS=OFF
cmake --build build/wasm
```

`build/wasm/web` then holds the module (`jkp_wasm.js`, `jkp_wasm.wasm`, 293 KB), the wrapper `jkp.js` and
the weights file. Serve them as static files; see [In the browser](browser.md).
