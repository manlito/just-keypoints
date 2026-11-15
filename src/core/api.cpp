#include <exception>
#include <stdexcept>
#include <memory>
#include <string>

#include "core/ops.h"
#include "core/weights.h"
#include "jkp/jkp.h"
#include "models/aliked/aliked.h"

struct jkp_ctx {
  jkp::Weights weights;
  std::unique_ptr<jkp::Aliked> model;
  std::unique_ptr<jkp::Ops> ops;
  jkp::DetectParams detect;
  jkp::Features features;
};

namespace {
thread_local std::string g_error;
}

extern "C" {

void jkp_config_default(jkp_config* config) {
  *config = jkp_config{};
  config->backend = JKP_BACKEND_CPU;
  config->model = "aliked-n16";
  config->top_k = -1;
  config->score_threshold = 0.2f;
  config->max_keypoints = 5000;
}

int jkp_backend_available(jkp_backend backend) {
#ifdef JKP_HAVE_CUDA
  if (backend == JKP_BACKEND_CUDA) return 1;
#endif
  return backend == JKP_BACKEND_CPU;
}

jkp_ctx* jkp_create(const jkp_config* config) {
  try {
    if (!config) throw std::runtime_error("config is null");
    std::string model = config->model ? config->model : "aliked-n16";
    auto ctx = std::make_unique<jkp_ctx>();
    const void* blob = nullptr;
    size_t blob_size = 0;
    if (config->weights_data && config->weights_size) {
      ctx->weights = jkp::Weights::from_memory(config->weights_data, config->weights_size);
    } else if (config->weights_path) {
      ctx->weights = jkp::Weights::from_file(config->weights_path);
    } else if (jkp::embedded_weights(model, &blob, &blob_size)) {
      ctx->weights = jkp::Weights::from_memory(blob, blob_size);
    } else {
      throw std::runtime_error("no weights given and '" + model + "' is not embedded in this build");
    }
    if (ctx->weights.meta("model").rfind("aliked-", 0) != 0)
      throw std::runtime_error("unsupported model '" + ctx->weights.meta("model") + "'");
    ctx->model = std::make_unique<jkp::Aliked>(ctx->weights);
    if (!jkp_backend_available(config->backend)) throw std::runtime_error("backend not available in this build");
    if (config->backend == JKP_BACKEND_CPU) ctx->ops = jkp::make_cpu_ops(config->num_threads);
#ifdef JKP_HAVE_CUDA
    if (config->backend == JKP_BACKEND_CUDA) ctx->ops = jkp::make_cuda_ops(config->device);
#endif
    ctx->detect.top_k = config->top_k;
    ctx->detect.scores_th = config->score_threshold;
    ctx->detect.n_limit = config->max_keypoints;
    return ctx.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

int jkp_extract(jkp_ctx* ctx, const jkp_image* image, jkp_result* result) {
  try {
    if (!ctx || !image || !result || !image->data) throw std::runtime_error("null argument");
    int channels = image->format == JKP_FORMAT_GRAY8 ? 1 : image->format == JKP_FORMAT_RGB8 ? 3 : 4;
    ctx->model->extract(*ctx->ops, image->data, image->width, image->height, image->stride, channels, ctx->detect,
                        &ctx->features);
    result->n = ctx->features.n;
    result->dim = ctx->features.dim;
    result->xy = ctx->features.xy.data();
    result->score = ctx->features.score.data();
    result->desc = ctx->features.desc.data();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

void jkp_destroy(jkp_ctx* ctx) { delete ctx; }

const char* jkp_last_error(void) { return g_error.c_str(); }

}  // extern "C"
