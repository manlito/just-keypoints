// The WebGPU backend on a device the caller created (jkp_config.webgpu_instance / webgpu_device), as an app
// does that runs other GPU work on the same device: results must equal those on the backend's own device,
// and the caller's device must still work after jkp_destroy.
//   webgpu_shared_test
#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "jkp/jkp.h"
#include "test_picture.h"

namespace {

void wait(WGPUInstance instance, WGPUFuture f) {
  WGPUFutureWaitInfo info{f, false};
  wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
}

bool extract(jkp_config config, const std::vector<uint8_t>& px, int w, int h, std::vector<float>* xy,
             std::vector<float>* desc) {
  config.top_k = 500;
  jkp_ctx* ctx = jkp_create(&config);
  if (!ctx) {
    std::fprintf(stderr, "jkp_create: %s\n", jkp_last_error());
    return false;
  }
  jkp_image image{px.data(), w, h, 0, JKP_FORMAT_RGB8};
  jkp_result r{};
  bool ok = jkp_extract(ctx, &image, &r) == 0 && jkp_extract(ctx, &image, &r) == 0;  // twice: pool reuse
  if (!ok) std::fprintf(stderr, "jkp_extract: %s\n", jkp_last_error());
  else xy->assign(r.xy, r.xy + 2 * r.n), desc->assign(r.desc, r.desc + static_cast<size_t>(r.n) * r.dim);
  jkp_destroy(ctx);
  return ok;
}

}  // namespace

int main() {
  WGPUInstanceDescriptor idesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
  const WGPUInstanceFeatureName feature = WGPUInstanceFeatureName_TimedWaitAny;
  idesc.requiredFeatureCount = 1;
  idesc.requiredFeatures = &feature;
  WGPUInstance instance = wgpuCreateInstance(&idesc);
  WGPUAdapter adapter = nullptr;
  WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  acb.mode = WGPUCallbackMode_WaitAnyOnly;
  acb.callback = [](WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView, void* u, void*) {
    if (st == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter*>(u) = a;
  };
  acb.userdata1 = &adapter;
  wait(instance, wgpuInstanceRequestAdapter(instance, nullptr, acb));
  if (!adapter) {
    std::fprintf(stderr, "no WebGPU adapter\n");
    return 1;
  }
  // The caller's device, with the limits the library would ask for itself and shader-f16.
  WGPULimits have = WGPU_LIMITS_INIT, want = WGPU_LIMITS_INIT;
  wgpuAdapterGetLimits(adapter, &have);
  want.maxStorageBufferBindingSize = have.maxStorageBufferBindingSize;
  want.maxBufferSize = have.maxBufferSize;
  WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  ddesc.requiredLimits = &want;
  const WGPUFeatureName f16 = WGPUFeatureName_ShaderF16;
  const bool has_f16 = wgpuAdapterHasFeature(adapter, f16);
  if (has_f16) ddesc.requiredFeatureCount = 1, ddesc.requiredFeatures = &f16;
  WGPUDevice device = nullptr;
  WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dcb.mode = WGPUCallbackMode_WaitAnyOnly;
  dcb.callback = [](WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView, void* u, void*) {
    if (st == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice*>(u) = d;
  };
  dcb.userdata1 = &device;
  wait(instance, wgpuAdapterRequestDevice(adapter, &ddesc, dcb));
  if (!device) {
    std::fprintf(stderr, "no WebGPU device\n");
    return 1;
  }

  const int w = 320, h = 240;
  const std::vector<uint8_t> px = picture(w, h, 3, w * 3, 31u);
  int failed = 0;
  for (jkp_precision precision : {JKP_PRECISION_FP32, JKP_PRECISION_FP16}) {
    if (precision == JKP_PRECISION_FP16 && !has_f16) continue;
    jkp_config own;
    jkp_config_default(&own);
    own.backend = JKP_BACKEND_WEBGPU;
    own.precision = precision;
    jkp_config shared = own;
    shared.webgpu_instance = instance;
    shared.webgpu_device = device;
    std::vector<float> xy_own, desc_own, xy_shared, desc_shared;
    const bool ok = extract(own, px, w, h, &xy_own, &desc_own) && extract(shared, px, w, h, &xy_shared, &desc_shared);
    const bool same = ok && xy_own == xy_shared && desc_own == desc_shared && !xy_own.empty();
    std::printf("%s: %zu keypoints on its own device, %zu on the caller's: %s\n",
                precision == JKP_PRECISION_FP16 ? "fp16" : "fp32", xy_own.size() / 2, xy_shared.size() / 2,
                same ? "identical" : "DIFFERENT");
    failed += !same;
  }
  // The caller's device is still usable after the library released its references.
  WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
  bd.size = 256;
  bd.usage = WGPUBufferUsage_CopyDst;
  WGPUBuffer b = wgpuDeviceCreateBuffer(device, &bd);
  const bool alive = b != nullptr;
  if (b) wgpuBufferRelease(b);
  std::printf("caller's device after jkp_destroy: %s\n", alive ? "usable" : "LOST");
  failed += !alive;
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuInstanceRelease(instance);
  std::printf(failed ? "FAILED (%d)\n" : "shared device ok\n", failed);
  return failed ? 1 : 0;
}
