# Accuracy and speed

## Agreement with the original

Every backend is checked against the original PyTorch ALIKED (`aliked-n16`) on sample images (31 for the CPU
backend, 57 for the GPU backends), in threshold and top-k mode. In FP32:

| Backend | Keypoint sets | Position error | Score error | Descriptor cosine |
|---|---|---|---|---|
| CPU | identical | <= 1.3e-4 px | <= 5e-5 | >= 1 - 5e-10 |
| CUDA | identical | <= 2.6e-4 px | <= 7e-5 | >= 1 - 2e-9 |
| Metal | identical | <= 2.6e-4 px | <= 8e-5 | >= 1 - 4e-9 |
| WebGPU (native) | identical | <= 2.4e-4 px | <= 7.6e-5 | >= 1 - 3.6e-9 |

Every intermediate tensor of the network is also compared with a reference forward pass that matches the
original; on every backend they agree within 3.1e-5. Keypoints whose scores nearly tie can come out in another
order on the GPU backends: the same set, in a different order.

### FP16

FP16 stores feature maps and weights as 16-bit floats. Against FP32 on the same backend: 98.7-100 % of the
keypoints are found within 1 px (mean 99.6-99.9 %), those move by at most 0.002 px (median), scores differ by at
most 0.04, and descriptors by about 2e-5 in cosine similarity on average. Against the original, at least 99.3 % of
the keypoints are the same (mean 99.8 %).

## Speed

Milliseconds per image, from an 8-bit image in host memory to keypoints and descriptors in host memory, with
1000 keypoints; median of warm runs.

| Size | Metal FP32 | Metal FP16 | PyTorch MPS | CUDA FP32 | CUDA FP16 | PyTorch CUDA |
|---|---|---|---|---|---|---|
| 640x480 | 8.3 | 7.2 | 13.1 | 15.2 | 13.4 | 15.6 |
| 1241x376 | 10.3 | 8.7 | 19.2 | 19.6 | 17.5 | 22.2 |
| 1024x768 | 13.8 | 11.3 | 28.9 | 26.8 | 24.5 | 31.8 |
| 1920x1080 | 30.2 | 23.3 | 74.8 | 59.1 | 53.9 | 77.3 |

Metal and MPS on an M5 Pro (macOS 26); CUDA on an RTX 2070 Max-Q laptop (CUDA 13.4, PyTorch 2.15). PyTorch runs
the original ALIKED. The CPU backend, for scale: 66 ms on the M5 Pro and 355 ms on the laptop at 640x480.

WebGPU, natively through Dawn on the same M5 Pro:

| Size | Metal FP32 | WebGPU FP32 | Metal FP16 | WebGPU FP16 |
|---|---|---|---|---|
| 640x480 | 8.4 | 10.1 | 7.3 | 8.8 |
| 1241x376 | 10.6 | 12.7 | 8.7 | 11.0 |
| 1024x768 | 13.9 | 17.1 | 11.2 | 15.2 |
| 1920x1080 | 29.7 | 38.0 | 23.2 | 33.3 |

In Chrome on the same machine: 12-13 ms at 640x480 in FP32 and 10 ms in FP16. <a href="bench/index.html">The benchmark
page</a> measures the same sizes in your browser, for ALIKED or RaCo.

For comparison, the ALIKED paper reports 12.9 ms for `aliked-n16` at 640x480 with 1000 keypoints (PyTorch,
RTX 2060).

## XFeat

Against the original `XFeat.detectAndCompute` on 59 images (320 px to 2522x818), top 4096 keypoints:

| Backend | Keypoint sets | Position error | Score error | Descriptor cosine |
|---|---|---|---|---|
| CPU | identical | 0 | <= 2.7e-5 | >= 1 - 5.4e-11 |
| Metal | identical | 0 | <= 2.7e-5 | >= 1 - 1.9e-10 |
| WebGPU (native) | identical | 0 | <= 5.3e-6 | >= 1 - 1.1e-10 |

Keypoints are pixels of the network's image, so a keypoint found by both is at exactly the same place. In FP16 on
Metal and WebGPU, 99.0 % or more of the keypoints are the same (mean 99.4 %). Milliseconds per image, top-k
1000, M5 Pro:

| Size | CPU | Metal FP32 | Metal FP16 | WebGPU FP32 | WebGPU FP16 | PyTorch MPS |
|---|---|---|---|---|---|---|
| 640x480 | 21.9 | 3.9 | 3.3 | 4.0 | 3.9 | 4.3 |
| 1241x376 | 28.2 | 4.3 | 3.6 | 4.4 | 4.2 | 4.3 |
| 1024x768 | 46.0 | 5.5 | 4.4 | 5.8 | 5.4 | 5.5 |
| 1920x1080 | 98.4 | 10.6 | 8.2 | 12.1 | 11.2 | 9.9 |

The CUDA backend does not run XFeat yet.
