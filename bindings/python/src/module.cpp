// Python bindings over the C API (include/jkp/jkp.h). Thin: the extractor and its
// options as they are in C, images in and results out as NumPy arrays (results are
// copies, so they outlive the next call). just_keypoints/__init__.py checks images.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "jkp/jkp.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Image = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;
using Out = nb::ndarray<nb::numpy, float>;

Out to_numpy(const float* src, std::vector<std::size_t> shape) {
  std::size_t size = 1;
  for (auto d : shape) size *= d;
  float* data = new float[std::max<std::size_t>(size, 1)];
  if (size) std::copy(src, src + size, data);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  return Out(data, shape.size(), shape.data(), owner);
}

const char* backend_name(jkp_backend b) {
  switch (b) {
    case JKP_BACKEND_CPU: return "cpu";
    case JKP_BACKEND_CUDA: return "cuda";
    case JKP_BACKEND_METAL: return "metal";
    case JKP_BACKEND_WEBGPU: return "webgpu";
  }
  return "?";
}

jkp_backend parse_backend(const std::string& s) {
  for (jkp_backend b : {JKP_BACKEND_CPU, JKP_BACKEND_CUDA, JKP_BACKEND_METAL, JKP_BACKEND_WEBGPU})
    if (s == backend_name(b)) return b;
  throw nb::value_error(("unknown backend '" + s + "'; one of cpu, cuda, metal, webgpu").c_str());
}

const char* const kBackends[] = {"cpu", "cuda", "metal", "webgpu"};

// Weights: a path (str or os.PathLike) or the bytes of a .jkpw file. Bytes are kept
// here for the life of the extractor.
struct WeightsSource {
  std::optional<std::string> path, data;
};

WeightsSource weights_source(nb::handle h, const char* name) {
  WeightsSource w;
  if (h.is_none()) return w;
  if (nb::isinstance<nb::bytes>(h)) {
    nb::bytes b = nb::borrow<nb::bytes>(h);
    w.data = std::string(b.c_str(), b.size());
  } else if (nb::isinstance<nb::str>(h) || nb::hasattr(h, "__fspath__")) {
    nb::object p = nb::module_::import_("os").attr("fspath")(h);
    w.path = nb::cast<std::string>(p);
  } else {
    throw nb::type_error((std::string(name) + " must be a path or bytes").c_str());
  }
  return w;
}

class Features {
 public:
  int n = 0, dim = 0;
  Out xy, score;
  std::optional<Out> desc, rank, cov;
};

class Extractor {
 public:
  Extractor(const std::string& backend, const std::string& model, nb::handle weights, int top_k,
            float score_threshold, int max_keypoints, int num_threads, int device,
            std::optional<std::string> descriptor, nb::handle descriptor_weights, bool skip_ranker,
            bool skip_covariance, const std::string& precision)
      : backend_(parse_backend(backend)), model_(model), descriptor_(std::move(descriptor)) {
    if (precision != "fp32" && precision != "fp16")
      throw nb::value_error(("unknown precision '" + precision + "'; fp32 or fp16").c_str());
    weights_ = weights_source(weights, "weights");
    descriptor_weights_ = weights_source(descriptor_weights, "descriptor_weights");
    jkp_config c;
    jkp_config_default(&c);
    c.backend = backend_;
    c.model = model_.c_str();
    if (weights_.data) {
      c.weights_data = weights_.data->data();
      c.weights_size = weights_.data->size();
    } else if (weights_.path) {
      c.weights_path = weights_.path->c_str();
    }
    c.top_k = top_k;
    c.score_threshold = score_threshold;
    c.max_keypoints = max_keypoints;
    c.num_threads = num_threads;
    c.device = device;
    if (descriptor_) c.descriptor = descriptor_->c_str();
    if (descriptor_weights_.data) {
      c.descriptor_weights_data = descriptor_weights_.data->data();
      c.descriptor_weights_size = descriptor_weights_.data->size();
    } else if (descriptor_weights_.path) {
      c.descriptor_weights_path = descriptor_weights_.path->c_str();
    }
    c.skip_ranker = skip_ranker;
    c.skip_covariance = skip_covariance;
    c.precision = precision == "fp16" ? JKP_PRECISION_FP16 : JKP_PRECISION_FP32;
    precision_ = precision;
    {
      nb::gil_scoped_release release;
      ctx_ = jkp_create(&c);
    }
    if (!ctx_) throw std::runtime_error(jkp_last_error());
  }
  ~Extractor() {
    if (ctx_) jkp_destroy(ctx_);
  }
  Extractor(const Extractor&) = delete;
  Extractor& operator=(const Extractor&) = delete;

  Features extract(const Image& image) {
    if (image.ndim() != 2 && image.ndim() != 3)
      throw nb::value_error("image must have shape (h, w), (h, w, 3) or (h, w, 4)");
    std::size_t ch = image.ndim() == 2 ? 1 : image.shape(2);
    if (ch != 1 && ch != 3 && ch != 4) throw nb::value_error("image must have 1, 3 or 4 channels");
    jkp_image im{image.data(), static_cast<int>(image.shape(1)), static_cast<int>(image.shape(0)), 0,
                 ch == 1 ? JKP_FORMAT_GRAY8 : ch == 3 ? JKP_FORMAT_RGB8 : JKP_FORMAT_RGBA8};
    if (im.width <= 0 || im.height <= 0) throw nb::value_error("image is empty");
    jkp_result r{};
    std::vector<float> xy, score, desc, rank, cov;
    bool has_desc, has_rank, has_cov;
    {
      nb::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);  // a context is not to be used from two threads at once
      if (jkp_extract(ctx_, &im, &r) != 0) throw std::runtime_error(jkp_last_error());
      // Copied under the lock: the arrays belong to the context until its next call.
      std::size_t n = r.n;
      xy.assign(r.xy, r.xy + 2 * n);
      score.assign(r.score, r.score + n);
      if ((has_desc = r.desc != nullptr)) desc.assign(r.desc, r.desc + n * r.dim);
      if ((has_rank = r.rank != nullptr)) rank.assign(r.rank, r.rank + n);
      if ((has_cov = r.cov != nullptr)) cov.assign(r.cov, r.cov + 4 * n);
    }
    std::size_t n = r.n, dim = r.dim;
    Features f;
    f.n = r.n;
    f.dim = r.dim;
    f.xy = to_numpy(xy.data(), {n, 2});
    f.score = to_numpy(score.data(), {n});
    if (has_desc) f.desc = to_numpy(desc.data(), {n, dim});
    if (has_rank) f.rank = to_numpy(rank.data(), {n});
    if (has_cov) f.cov = to_numpy(cov.data(), {n, 2, 2});
    return f;
  }

  std::string backend() const { return backend_name(backend_); }
  const std::string& model() const { return model_; }
  const std::string& precision() const { return precision_; }
  std::optional<std::string> descriptor() const { return descriptor_; }

 private:
  jkp_backend backend_;
  std::string model_, precision_;
  std::optional<std::string> descriptor_;
  WeightsSource weights_, descriptor_weights_;
  jkp_ctx* ctx_ = nullptr;
  std::mutex mutex_;
};

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "just-keypoints: ALIKED and RaCo keypoints with minimal dependencies (the compiled core).";

  m.def(
      "backend_available", [](const std::string& b) { return jkp_backend_available(parse_backend(b)) != 0; },
      "backend"_a,
      "Whether the backend ('cpu', 'cuda', 'metal' or 'webgpu') is compiled into this build. Whether this "
      "machine can run it is known when an Extractor is made.");
  m.def(
      "compiled_backends",
      [] {
        std::vector<std::string> out;
        for (const char* b : kBackends)
          if (jkp_backend_available(parse_backend(b))) out.push_back(b);
        return out;
      },
      "The backends this build has: 'cpu' always, 'metal' in the macOS package, 'cuda' in "
      "just-keypoints-cu12 and just-keypoints-cu13.");

  nb::class_<Features>(m, "Features", "Keypoints of one image. Arrays are float32 copies.")
      .def_ro("n", &Features::n, "Number of keypoints.")
      .def_ro("dim", &Features::dim, "Descriptor size: 128 for aliked-n16, 64 for xfeat, 0 for raco without a descriptor model.")
      .def_prop_ro("xy", [](const Features& f) { return f.xy; }, nb::rv_policy::move,
              "(n, 2) pixel coordinates, x right, y down, (0, 0) the centre of the top-left pixel.")
      .def_prop_ro("score", [](const Features& f) { return f.score; }, nb::rv_policy::move,
              "(n,) aliked: in [0, 1]. raco: a probability, summing to 1 over the image. xfeat: in (0, 1).")
      .def_prop_ro("desc", [](const Features& f) { return f.desc; }, nb::rv_policy::move, "(n, dim) L2-normalized descriptors, or None when dim is 0.")
      .def_prop_ro("rank", [](const Features& f) { return f.rank; }, nb::rv_policy::move, "(n,) ranker score, higher is better (raco), or None.")
      .def_prop_ro("cov", [](const Features& f) { return f.cov; }, nb::rv_policy::move, "(n, 2, 2) keypoint covariance in pixels squared (raco), or None.")
      .def("__len__", [](const Features& f) { return f.n; })
      .def("__repr__", [](const Features& f) {
        return "Features(n=" + std::to_string(f.n) + ", dim=" + std::to_string(f.dim) + ")";
      });

  nb::class_<Extractor>(m, "Extractor",
                        "A model on a backend, ready to extract keypoints. Made once, used for many images. "
                        "Calls from several threads are serialized.")
      .def(nb::init<const std::string&, const std::string&, nb::handle, int, float, int, int, int,
                    std::optional<std::string>, nb::handle, bool, bool, const std::string&>(),
           nb::kw_only(), "backend"_a = "cpu", "model"_a = "aliked-n16", "weights"_a = nb::none(),
           "top_k"_a = -1, "score_threshold"_a = 0.2f, "max_keypoints"_a = 5000, "num_threads"_a = 0,
           "device"_a = 0, "descriptor"_a = nb::none(), "descriptor_weights"_a = nb::none(),
           "skip_ranker"_a = false, "skip_covariance"_a = false, "precision"_a = "fp32",
           "backend: 'cpu', 'cuda', 'metal' or 'webgpu'.\n"
           "model: 'aliked-n16' (keypoints and descriptors), 'raco' (keypoints with ranker scores and "
           "covariances, no descriptors) or 'xfeat' (a light detector with 64-dim descriptors).\n"
           "weights: a .jkpw path or its bytes; None uses the weights built into the package.\n"
           "top_k > 0 keeps the top_k best keypoints; otherwise every keypoint with score > score_threshold, "
           "at most max_keypoints. raco: always the top_k best (2048 when top_k <= 0); xfeat: the top_k best "
           "above its own threshold (4096 when top_k <= 0).\n"
           "num_threads: CPU worker threads, 0 = all. device: GPU index.\n"
           "descriptor: 'aliked-n16' describes raco's keypoints (descriptor_weights as weights).\n"
           "skip_ranker, skip_covariance: raco leaves out that head (its output is None).\n"
           "precision: 'fp32' (the reference) or 'fp16' (GPU backends, aliked and xfeat).\n"
           "Raises RuntimeError when the backend, device, model or weights cannot be used.")
      .def("_extract", &Extractor::extract, "image"_a.noconvert())
      .def_prop_ro("backend", &Extractor::backend)
      .def_prop_ro("model", &Extractor::model)
      .def_prop_ro("descriptor", &Extractor::descriptor)
      .def_prop_ro("precision", &Extractor::precision);
}
