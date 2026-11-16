// Defaults for operations a backend may not have yet.
#include "core/ops.h"

#include <stdexcept>

namespace jkp {
namespace {
[[noreturn]] void unsupported() { throw std::runtime_error("RaCo is not implemented on this backend yet"); }
}  // namespace

TensorPtr Ops::normalize(const TensorPtr&, const float*, const float*) { unsupported(); }
TensorPtr Ops::pad_reflect(const TensorPtr&, int) { unsupported(); }
void Ops::detect_raco(const TensorPtr&, const TensorPtr&, const TensorPtr&, const RacoDetectParams&,
                      RacoKeypoints*) {
  unsupported();
}

}  // namespace jkp
