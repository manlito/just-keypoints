// Backend by name, for the tests. Throws when the backend is not in this build.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "core/ops.h"

inline std::unique_ptr<jkp::Ops> make_backend(const std::string& name) {
  if (name == "cpu") return jkp::make_cpu_ops(0);
#ifdef JKP_TEST_CUDA
  if (name == "cuda") return jkp::make_cuda_ops(0);
#endif
#ifdef JKP_TEST_METAL
  if (name == "metal") return jkp::make_metal_ops(0);
#endif
  throw std::runtime_error("this build has no " + name + " backend");
}
