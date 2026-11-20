#include <exception>
#include <stdexcept>
#include <memory>
#include <string>

#include "core/ops.h"
#include "core/weights.h"
#include "jkp/jkp.h"
#include "models/aliked/aliked.h"
#include "models/raco/raco.h"
#include "models/xfeat/xfeat.h"

struct jkp_ctx {
  jkp::Weights weights, descriptor_weights;
  std::unique_ptr<jkp::Aliked> model;  // one of model / raco is set
  std::unique_ptr<jkp::Aliked> descriptor;  // optional, describes the keypoints of raco
  std::unique_ptr<jkp::Raco> raco;
  std::unique_ptr<jkp::Xfeat> xfeat;
  jkp::RacoOptions raco_options;
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
#ifdef JKP_HAVE_METAL
  if (backend == JKP_BACKEND_METAL) return 1;
#endif
#ifdef JKP_HAVE_WEBGPU
  if (backend == JKP_BACKEND_WEBGPU) return 1;
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
    const std::string& kind = ctx->weights.meta("model");
    if (kind == "raco") ctx->raco = std::make_unique<jkp::Raco>(ctx->weights);
    else if (kind == "xfeat") ctx->xfeat = std::make_unique<jkp::Xfeat>(ctx->weights);
    else if (kind.rfind("aliked-", 0) == 0) ctx->model = std::make_unique<jkp::Aliked>(ctx->weights);
    else throw std::runtime_error("unsupported model '" + kind + "'");
    ctx->raco_options.top_k = config->top_k;
    ctx->raco_options.ranker = !config->skip_ranker;
    ctx->raco_options.covariance = !config->skip_covariance;
    if (config->descriptor && *config->descriptor && !((ctx->model || ctx->xfeat) && kind == config->descriptor)) {
      std::string name = config->descriptor;
      if (!ctx->raco) throw std::runtime_error("'" + kind + "' has its own descriptors; a descriptor model needs raco");
      if (config->descriptor_weights_data && config->descriptor_weights_size) {
        ctx->descriptor_weights =
            jkp::Weights::from_memory(config->descriptor_weights_data, config->descriptor_weights_size);
      } else if (config->descriptor_weights_path) {
        ctx->descriptor_weights = jkp::Weights::from_file(config->descriptor_weights_path);
      } else if (jkp::embedded_weights(name, &blob, &blob_size)) {
        ctx->descriptor_weights = jkp::Weights::from_memory(blob, blob_size);
      } else {
        throw std::runtime_error("no descriptor weights given and '" + name + "' is not embedded in this build");
      }
      if (ctx->descriptor_weights.meta("model").rfind("aliked-", 0) != 0)
        throw std::runtime_error("unsupported descriptor model '" + ctx->descriptor_weights.meta("model") + "'");
      ctx->descriptor = std::make_unique<jkp::Aliked>(ctx->descriptor_weights);
    }
    if (!jkp_backend_available(config->backend)) throw std::runtime_error("backend not available in this build");
    if (config->precision != JKP_PRECISION_FP32 && config->precision != JKP_PRECISION_FP16)
      throw std::runtime_error("unknown precision");
    const jkp::Precision precision =
        config->precision == JKP_PRECISION_FP16 ? jkp::Precision::Fp16 : jkp::Precision::Fp32;
    if (precision == jkp::Precision::Fp16 && config->backend == JKP_BACKEND_CPU)
      throw std::runtime_error("the cpu backend runs in fp32 only");
    if (precision == jkp::Precision::Fp16 && ctx->raco)
      throw std::runtime_error("raco runs in fp32 only");
    if (config->backend == JKP_BACKEND_CPU) ctx->ops = jkp::make_cpu_ops(config->num_threads);
#ifdef JKP_HAVE_CUDA
    if (config->backend == JKP_BACKEND_CUDA) ctx->ops = jkp::make_cuda_ops(config->device, precision);
#endif
#ifdef JKP_HAVE_METAL
    if (config->backend == JKP_BACKEND_METAL) ctx->ops = jkp::make_metal_ops(config->device, precision);
#endif
#ifdef JKP_HAVE_WEBGPU
    if (config->backend == JKP_BACKEND_WEBGPU)
      ctx->ops = jkp::make_webgpu_ops(precision, config->webgpu_instance, config->webgpu_device);
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
    if (ctx->raco)
      ctx->raco->extract(*ctx->ops, image->data, image->width, image->height, image->stride, channels,
                         ctx->raco_options, &ctx->features);
    if (ctx->descriptor) {
      ctx->descriptor->describe_at(*ctx->ops, image->data, image->width, image->height, image->stride, channels,
                                   ctx->features.xy.data(), ctx->features.n, &ctx->features.desc);
      ctx->features.dim = ctx->descriptor->descriptor_dim();
    }
    if (ctx->xfeat)
      ctx->xfeat->extract(*ctx->ops, image->data, image->width, image->height, image->stride, channels,
                          ctx->detect.top_k, &ctx->features);
    else if (!ctx->raco)
      ctx->model->extract(*ctx->ops, image->data, image->width, image->height, image->stride, channels, ctx->detect,
                          &ctx->features);
    result->n = ctx->features.n;
    result->dim = ctx->features.dim;
    result->xy = ctx->features.xy.data();
    result->score = ctx->features.score.data();
    result->desc = ctx->features.desc.empty() ? nullptr : ctx->features.desc.data();
    result->rank = ctx->features.rank.empty() ? nullptr : ctx->features.rank.data();
    result->cov = ctx->features.cov.empty() ? nullptr : ctx->features.cov.data();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

void jkp_destroy(jkp_ctx* ctx) { delete ctx; }

const char* jkp_last_error(void) { return g_error.c_str(); }

}  // extern "C"
