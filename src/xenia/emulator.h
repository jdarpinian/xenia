/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_EMULATOR_H_
#define XENIA_EMULATOR_H_

#include <string>

#include <xenia/common.h>
#include <xenia/core.h>
#include <xenia/debug_agent.h>
#include <xenia/kernel/kernel_state.h>
#include <xenia/xbox.h>

namespace xe {
namespace apu {
class AudioSystem;
}  // namespace apu
namespace cpu {
class Processor;
}  // namespace cpu
namespace gpu {
class GraphicsSystem;
}  // namespace gpu
namespace hid {
class InputSystem;
}  // namespace hid
namespace kernel {
class XamModule;
class XboxkrnlModule;
}  // namespace kernel
namespace ui {
class Window;
}  // namespace ui
}  // namespace xe

namespace xe {

class ExportResolver;

class Emulator {
 public:
  Emulator(const std::wstring& command_line);
  ~Emulator();

  const std::wstring& command_line() const { return command_line_; }

  ui::Window* main_window() const { return main_window_; }
  void set_main_window(ui::Window* window);

  Memory* memory() const { return memory_; }

  DebugAgent* debug_agent() const { return debug_agent_.get(); }

  cpu::Processor* processor() const { return processor_; }
  apu::AudioSystem* audio_system() const { return audio_system_; }
  gpu::GraphicsSystem* graphics_system() const { return graphics_system_; }
  hid::InputSystem* input_system() const { return input_system_; }

  ExportResolver* export_resolver() const { return export_resolver_; }
  kernel::fs::FileSystem* file_system() const { return file_system_; }

  kernel::XboxkrnlModule* xboxkrnl() const { return xboxkrnl_; }
  kernel::XamModule* xam() const { return xam_; }

  X_STATUS Setup();

  // TODO(benvanik): raw binary.
  X_STATUS LaunchXexFile(const std::wstring& path);
  X_STATUS LaunchDiscImage(const std::wstring& path);
  X_STATUS LaunchSTFSTitle(const std::wstring& path);

 private:
  X_STATUS CompleteLaunch(const std::wstring& path,
                          const std::string& module_path);

  std::wstring command_line_;

  ui::Window* main_window_;

  Memory* memory_;

  std::unique_ptr<DebugAgent> debug_agent_;

  cpu::Processor* processor_;
  apu::AudioSystem* audio_system_;
  gpu::GraphicsSystem* graphics_system_;
  hid::InputSystem* input_system_;

  ExportResolver* export_resolver_;
  kernel::fs::FileSystem* file_system_;

  kernel::KernelState* kernel_state_;
  kernel::XamModule* xam_;
  kernel::XboxkrnlModule* xboxkrnl_;
};

}  // namespace xe

#endif  // XENIA_EMULATOR_H_
