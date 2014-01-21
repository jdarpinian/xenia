/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/gpu/gl/gl_window.h>

#include <GL/glew.h>
#include <GL/wglew.h>

using namespace xe;
using namespace xe::core;
using namespace xe::gpu;
using namespace xe::gpu::gl;


GLWindow::GLWindow(
    xe_run_loop_ref run_loop) :
    Win32Window(run_loop) {
  hdc_ = GetDC(handle());
}

GLWindow::~GLWindow() {
}

void GLWindow::Swap() {
  // Swap buffers.
  SwapBuffers(hdc_);
}

void GLWindow::OnResize(uint32_t width, uint32_t height) {
  Win32Window::OnResize(width, height);

  // TODO(benvanik): resize swap buffers?
}

void GLWindow::OnClose() {
  // We are the master window - if they close us, quit!
  xe_run_loop_quit(run_loop_);
  exit(0);
}

HDC GLWindow::hdc() {
	return hdc_;
}