#pragma once

#include <memory>
#include <string>

namespace nn {

// Visible CUDA devices. Context() leaves the current device alone.
int device_count();
std::string device_name(int device);

class Context {
 public:
  Context();
  // Calls cudaSetDevice. Throws device when the index is outside 0 .. device_count()-1.
  explicit Context(int device);
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
