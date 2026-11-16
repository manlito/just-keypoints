// Flat entry points of the WebAssembly module, over the C API (include/jkp/jkp.h). web/jkp.js wraps
// them; an app that links just-keypoints into its own module calls the C API directly instead.
//
// jkpw_create and jkpw_extract wait for the GPU, so they are JSPI exports: in JavaScript they return
// promises, and the module suspends at each wait instead of blocking the page.
#include <emscripten/emscripten.h>

#include <cstdint>
#include <new>

#include "jkp/jkp.h"

namespace {
struct Handle {
  jkp_ctx* ctx = nullptr;
  jkp_result result{};
};
}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int jkpw_backend_available(int backend) {
  return jkp_backend_available(static_cast<jkp_backend>(backend));
}

// weights: a .jkpw file in module memory, copied by jkp_create. Returns null on failure (jkpw_error).
EMSCRIPTEN_KEEPALIVE Handle* jkpw_create(int backend, int precision, const void* weights, int weights_size, int top_k,
                                         float score_threshold, int max_keypoints) {
  jkp_config config;
  jkp_config_default(&config);
  config.backend = static_cast<jkp_backend>(backend);
  config.precision = static_cast<jkp_precision>(precision);
  config.weights_data = weights;
  config.weights_size = static_cast<size_t>(weights_size);
  config.top_k = top_k;
  config.score_threshold = score_threshold;
  config.max_keypoints = max_keypoints;
  jkp_ctx* ctx = jkp_create(&config);
  if (!ctx) return nullptr;
  Handle* h = new (std::nothrow) Handle;
  if (!h) {
    jkp_destroy(ctx);
    return nullptr;
  }
  h->ctx = ctx;
  return h;
}

// rgba: width * height * 4 bytes. Returns the number of keypoints, or -1 (jkpw_error).
EMSCRIPTEN_KEEPALIVE int jkpw_extract(Handle* h, const uint8_t* rgba, int width, int height) {
  jkp_image image{rgba, width, height, 0, JKP_FORMAT_RGBA8};
  if (jkp_extract(h->ctx, &image, &h->result) != 0) return -1;
  return h->result.n;
}

EMSCRIPTEN_KEEPALIVE int jkpw_dim(Handle* h) { return h->result.dim; }
EMSCRIPTEN_KEEPALIVE const float* jkpw_xy(Handle* h) { return h->result.xy; }
EMSCRIPTEN_KEEPALIVE const float* jkpw_score(Handle* h) { return h->result.score; }
EMSCRIPTEN_KEEPALIVE const float* jkpw_desc(Handle* h) { return h->result.desc; }
EMSCRIPTEN_KEEPALIVE const char* jkpw_error() { return jkp_last_error(); }

EMSCRIPTEN_KEEPALIVE void jkpw_destroy(Handle* h) {
  if (!h) return;
  jkp_destroy(h->ctx);
  delete h;
}

}  // extern "C"
