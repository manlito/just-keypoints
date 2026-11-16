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

typedef struct {
  jkp_backend backend;
  /* Model to run; selects the embedded weights when no weights are given. Default "aliked-n16". */
  const char* model;
  /* Weights source, first that is set wins: memory blob, file path, weights embedded at build time. */
  const void* weights_data;
  size_t weights_size;
  const char* weights_path;
  /* Selection. top_k > 0: keep the top_k best keypoints. Otherwise keep every keypoint with
     score > score_threshold, capped at max_keypoints (best first when over the cap). */
  int top_k;
  float score_threshold;
  int max_keypoints;
  /* CPU backend worker threads; 0 = all hardware threads. */
  int num_threads;
  /* GPU backends: device index; 0 = first device. */
  int device;
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
  int dim;            /* descriptor size (128 for aliked-n16) */
  const float* xy;    /* [n][2] pixel coordinates, x right / y down, (0,0) = centre of the top-left pixel */
  const float* score; /* [n] */
  const float* desc;  /* [n][dim], L2-normalized */
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
