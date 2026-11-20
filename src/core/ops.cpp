// Defaults for operations a backend may not have yet.
#include "core/ops.h"

#include <stdexcept>

namespace jkp {
namespace {
[[noreturn]] void unsupported() { throw std::runtime_error("RaCo is not implemented on this backend yet"); }
[[noreturn]] void unsupported_xfeat() { throw std::runtime_error("XFeat is not implemented on this backend yet"); }
}  // namespace

TensorPtr Ops::resize(const TensorPtr&, int, int) { unsupported_xfeat(); }
TensorPtr Ops::gray_instance_norm(const TensorPtr&) { unsupported_xfeat(); }
TensorPtr Ops::conv2d_stride2(const TensorPtr&, const Param&, const Param*, Act) { unsupported_xfeat(); }
TensorPtr Ops::space_to_depth(const TensorPtr&, int) { unsupported_xfeat(); }
void Ops::detect_xfeat(const TensorPtr&, const TensorPtr&, const TensorPtr&, const XfeatDetectParams&,
                       XfeatKeypoints*) {
  unsupported_xfeat();
}

TensorPtr Ops::normalize(const TensorPtr&, const float*, const float*) { unsupported(); }
TensorPtr Ops::pad_reflect(const TensorPtr&, int) { unsupported(); }
void Ops::detect_raco(const TensorPtr&, const TensorPtr&, const TensorPtr&, const RacoDetectParams&,
                      RacoKeypoints*) {
  unsupported();
}

}  // namespace jkp
