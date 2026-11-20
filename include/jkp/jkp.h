/* just-keypoints: keypoint + descriptor extraction with minimal dependencies. */
#ifndef JKP_H
#define JKP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jkp_ctx jkp_ctx;

typedef enum { JKP_BACKEND_CPU = 0, JKP_BACKEND_CUDA = 1, JKP_BACKEND_METAL = 2, JKP_BACKEND_WEBGPU = 3 } jkp_backend;

typedef enum { JKP_FORMAT_GRAY8 = 0, JKP_FORMAT_RGB8 = 1, JKP_FORMAT_RGBA8 = 2 } jkp_format;

/* FP32 is the reference mode: it matches the original network. FP16 (GPU backends only) stores feature
   maps and weights as 16-bit floats: faster, with slightly different keypoints and descriptors. */
typedef enum { JKP_PRECISION_FP32 = 0, JKP_PRECISION_FP16 = 1 } jkp_precision;

typedef struct {
  jkp_backend backend;
  /* Model to run; selects the embedded weights when no weights are given. Default "aliked-n16".
     "raco" is a detector without descriptors: it returns ranker scores and covariances instead.
     "xfeat" is a light detector with 64-dim descriptors (about a third of aliked-n16's time on the CPU). */
  const char* model;
  /* Weights source, first that is set wins: memory blob, file path, weights embedded at build time. */
  const void* weights_data;
  size_t weights_size;
  const char* weights_path;
  /* Selection. top_k > 0: keep the top_k best keypoints. Otherwise keep every keypoint with
     score > score_threshold, capped at max_keypoints (best first when over the cap).
     raco: always the top_k best (2048 when top_k <= 0); score_threshold and max_keypoints are not used.
     xfeat: the top_k best above XFeat's own detection threshold (4096 when top_k <= 0); score_threshold and
     max_keypoints are not used. The network sees the image resized down to a multiple of 32 (at least 32 x 32). */
  int top_k;
  float score_threshold;
  int max_keypoints;
  /* CPU backend worker threads; 0 = all hardware threads. */
  int num_threads;
  /* GPU backends: device index; 0 = first device. */
  int device;
  /* Optional second model that describes the keypoints of a detector without descriptors:
     model = "raco", descriptor = "aliked-n16". NULL = only what `model` itself returns.
     Its weights: memory blob, file path, or weights embedded at build time, as above. */
  const char* descriptor;
  const void* descriptor_weights_data;
  size_t descriptor_weights_size;
  const char* descriptor_weights_path;
  /* raco: non-zero skips a head and leaves its output NULL. The covariance head is about three
     quarters of the work of the network, the ranker head about a fifth. */
  int skip_ranker;
  int skip_covariance;
  /* Default JKP_PRECISION_FP32. jkp_create fails when the backend does not support the precision. */
  jkp_precision precision;
  /* WebGPU backend: a WGPUInstance and a WGPUDevice created from it, to run on the caller's device
     (an app that also runs other GPU work shares one device). Both NULL: the backend creates its own.
     The device must outlive the context; it must not be used from another thread during jkp_extract.
     fp16 needs the device to have the shader-f16 feature. `device` above is not used by WebGPU. */
  void* webgpu_instance;
  void* webgpu_device;
} jkp_config;

typedef struct {
  const uint8_t* data; /* top-left origin, rows of `stride` bytes */
  int width;
  int height;
  int stride; /* bytes per row; 0 = width * bytes per pixel */
  jkp_format format;
} jkp_image;

/* Arrays are owned by the context and stay valid until the next jkp_extract or jkp_destroy on it. */
typedef struct {
  int n;              /* number of keypoints */
  int dim;            /* descriptor size (128 for aliked-n16, 64 for xfeat, 0 for raco without a descriptor model) */
  const float* xy;    /* [n][2] pixel coordinates, x right / y down, (0,0) = centre of the top-left pixel */
  const float* score; /* [n]; aliked: in [0, 1]. raco: probability, sums to 1 over the image. xfeat: in (0, 1) */
  const float* desc;  /* [n][dim], L2-normalized; NULL when dim is 0 */
  const float* rank;  /* [n] ranker score, higher is better (raco); NULL for other models */
  const float* cov;   /* [n][2][2] keypoint covariance in pixels^2 (raco); NULL for other models */
} jkp_result;

/* Fills `config` with defaults (CPU, aliked-n16, threshold 0.2, at most 5000 keypoints). */
void jkp_config_default(jkp_config* config);

/* Returns 1 when the backend is compiled into this build (the device itself is checked by jkp_create). */
int jkp_backend_available(jkp_backend backend);

/* Returns NULL on failure; see jkp_last_error(). */
jkp_ctx* jkp_create(const jkp_config* config);

/* Returns 0 on success. A context must not be used from several threads at once. */
int jkp_extract(jkp_ctx* ctx, const jkp_image* image, jkp_result* result);

void jkp_destroy(jkp_ctx* ctx);

/* Message for the last failure on the calling thread. */
const char* jkp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* JKP_H */
