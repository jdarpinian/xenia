/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_WINDOW_H_
#define XENIA_GPU_GL_GL_WINDOW_H_

#include <xenia/core.h>
#include <xenia/core/win32_window.h>

namespace xe {
namespace gpu {
namespace gl {


class GLWindow : public xe::core::Win32Window {
public:
  GLWindow(
      xe_run_loop_ref run_loop);
  virtual ~GLWindow();

  void Swap();
  HDC hdc();

protected:
  virtual void OnResize(uint32_t width, uint32_t height);
  virtual void OnClose();

private:
	HDC hdc_;
};


}  // namespace gl
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_WINDOW_H_
