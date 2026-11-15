// Reader/writer for the .jkpw / .jkpt tensor container. Format: see tools/jkpfile.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace jkp {

// A float32 tensor that lives inside a loaded container.
struct Param {
  std::vector<int> shape;
  const float* data = nullptr;
  size_t size() const {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return n;
  }
};

class Weights {
 public:
  // Both throw std::runtime_error on malformed input. The blob is copied.
  static Weights from_memory(const void* data, size_t size);
  static Weights from_file(const std::string& path);

  const Param& get(const std::string& name) const;       // throws if missing
  const Param* find(const std::string& name) const;      // nullptr if missing
  const std::string& meta(const std::string& key) const; // throws if missing
  int meta_int(const std::string& key) const;

 private:
  std::vector<uint32_t> storage_;  // 4-byte aligned copy of the file
  std::map<std::string, Param> tensors_;
  std::map<std::string, std::string> meta_;
};

// Tensor to be written by write_container (float32 only).
struct NamedTensor {
  std::string name;
  std::vector<int> shape;
  const float* data;
};
void write_container(const std::string& path, const std::map<std::string, std::string>& meta,
                     const std::vector<NamedTensor>& tensors);

// Weights compiled into the library (JKP_EMBED_WEIGHTS). Returns false when `model` is not embedded.
bool embedded_weights(const std::string& model, const void** data, size_t* size);

}  // namespace jkp
