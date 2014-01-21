/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/gpu/gl/gl_gpu.h>

#include <xenia/gpu/gl/gl_graphics_system.h>


using namespace xe;
using namespace xe::gpu;
using namespace xe::gpu::gl;


namespace {
  void InitializeIfNeeded();
  void CleanupOnShutdown();

  void InitializeIfNeeded() {
    static bool has_initialized = false;
    if (has_initialized) {
      return;
    }
    has_initialized = true;

    //

    atexit(CleanupOnShutdown);
  }

  void CleanupOnShutdown() {
  }
}


GraphicsSystem* xe::gpu::gl::Create(Emulator* emulator) {
  InitializeIfNeeded();
  return new GLGraphicsSystem(emulator);
}
