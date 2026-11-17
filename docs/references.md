# References and acknowledgements

just-keypoints runs networks other people designed and trained, and it is checked against their code. This page
credits that work, the images the demo uses, and the software the library and this site are built with.

## Networks

### ALIKED

Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C. Y. Chen, Qingsong Xu and Zhengguo Li. "ALIKED: A Lighter
Keypoint and Descriptor Extraction Network via Deformable Transformation." IEEE Transactions on
Instrumentation and Measurement, 2023. <https://github.com/Shiaoming/ALIKED>. BSD 3-Clause licence.

- The `aliked-n16` weights are the original checkpoint, with BatchNorm folded into the convolutions.
- The network, the keypoint detection (DKD) and the descriptor head (SDDH) follow the original code, and every
  backend is checked against it.

### RaCo

Abhiram Shenoi, Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys. "RaCo: Ranking and Covariance for
Practical Learned Keypoints." 3DV 2026. <https://github.com/cvg/RaCo>. Apache 2.0 licence.

- The RaCo weights are the released checkpoint, with BatchNorm folded.
- Describing RaCo keypoints with ALIKED follows `ALIKED.describe` in LightGlue
  (<https://github.com/cvg/LightGlue>, Apache 2.0).

## Images in the demo

- The indoor pair: two frames of the TUM RGB-D benchmark. Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
  Burgard and Daniel Cremers. "A Benchmark for the Evaluation of RGB-D SLAM Systems." IROS 2012.
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>. CC BY 4.0.
- The street pair: two frames of the KITTI odometry benchmark. Andreas Geiger, Philip Lenz and Raquel Urtasun.
  "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR 2012.
  <https://www.cvlibs.net/datasets/kitti/>. CC BY-NC-SA 3.0.

Both pairs are among the sample images of the ALIKED repository.

## Software

- **Dawn** and **emdawnwebgpu**, Google's WebGPU implementation and its Emscripten bindings: the WebGPU backend,
  natively and in the browser. <https://dawn.googlesource.com/dawn>. BSD 3-Clause licence.
- **Emscripten**: the browser build. <https://emscripten.org>. MIT and University of Illinois/NCSA licences.
- **stb_image** by Sean Barrett: image decoding in the command-line tool only. Public domain or MIT.
- **PyTorch**: running the original networks when checking results. It is never a dependency of the library.
- **Sphinx** and **MyST-Parser**: this site. The theme follows the documentation site of just-ba, a bundle
  adjustment library by the same authors.
- **Inter** and **IBM Plex Mono**, from Google Fonts: the site's typefaces. SIL Open Font License.
