// Backend by name, for the tests: cpu, cuda, metal, or a GPU backend in FP16 as "cuda:fp16" / "metal:fp16".
// Throws when the backend is not in this build.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "core/ops.h"

inline std::unique_ptr<jkp::Ops> make_backend(const std::string& spec) {
  const size_t colon = spec.find(':');
  const std::string name = spec.substr(0, colon), mode = colon == std::string::npos ? "fp32" : spec.substr(colon + 1);
  if (mode != "fp32" && mode != "fp16") throw std::runtime_error("unknown precision '" + mode + "'");
  const jkp::Precision precision = mode == "fp16" ? jkp::Precision::Fp16 : jkp::Precision::Fp32;
  if (name == "cpu" && precision == jkp::Precision::Fp32) return jkp::make_cpu_ops(0);
#ifdef JKP_TEST_CUDA
  if (name == "cuda") return jkp::make_cuda_ops(0, precision);
#endif
#ifdef JKP_TEST_METAL
  if (name == "metal") return jkp::make_metal_ops(0, precision);
#endif
  (void)precision;
  throw std::runtime_error("this build has no " + spec + " backend");
}
