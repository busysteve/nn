#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nn {

struct DeviceInfo {
  int index;
  std::string name;
};

// Visible CUDA devices. Context() leaves the current device alone.
int device_count();
std::string device_name(int device);
// Same devices as device_count(), in index order.
std::vector<DeviceInfo> detect_devices();
// "auto" is the first detected device. Any other choice is a decimal index.
// Throws device when nothing was detected, the text is neither, or the index is outside the list.
int select_device(const std::string& choice);

class Context {
 public:
  Context();
  // Calls cudaSetDevice. Throws device when the index is outside 0 .. device_count()-1.
  explicit Context(int device);
  // "auto" or a decimal index. Same rules as select_device, then cudaSetDevice.
  explicit Context(const std::string& choice);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  void synchronize() const;

 private:
  void open_handles();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend struct ContextAccess;
};

}  // namespace nn
