/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/gpu/gl/gl_graphics_system.h>

#include <xenia/gpu/gpu-private.h>
#include <xenia/gpu/gl/gl_graphics_driver.h>
#include <xenia/gpu/gl/gl_window.h>


using namespace xe;
using namespace xe::gpu;
using namespace xe::gpu::gl;


namespace {

void __stdcall GLGraphicsSystemVsyncCallback(
    GLGraphicsSystem* gs, BOOLEAN) {
  gs->MarkVblank();
  gs->DispatchInterruptCallback(0);
}

}


GLGraphicsSystem::GLGraphicsSystem(Emulator* emulator) :
    window_(0), timer_queue_(NULL),
    vsync_timer_(NULL), GraphicsSystem(emulator) {
}

GLGraphicsSystem::~GLGraphicsSystem() {
}

static void APIENTRY GLDebugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam) {
  XELOGGPU("GL debug message: %s", message);
}

void GLGraphicsSystem::Initialize() {
  GraphicsSystem::Initialize();

  XEASSERTNULL(timer_queue_);
  XEASSERTNULL(vsync_timer_);

  timer_queue_ = CreateTimerQueue();
  CreateTimerQueueTimer(
      &vsync_timer_,
      timer_queue_,
      (WAITORTIMERCALLBACK)GLGraphicsSystemVsyncCallback,
      this,
      16,
      100,
      WT_EXECUTEINTIMERTHREAD);

  // Create the window.
  // This will pump through the run-loop and and be where our swapping
  // will take place.
  XEASSERTNULL(window_);
  window_ = new GLWindow(run_loop_);
  window_->set_title(XETEXT("Xenia OpenGL"));

  // Initialize OpenGL and GLEW.
  // Do the Windows pixel format selection dance.
  PIXELFORMATDESCRIPTOR pfd;
  memset(&pfd, 0, sizeof(pfd));
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;
  pfd.cDepthBits = 24;
  pfd.cStencilBits = 8;
  HDC hdc = window_->hdc();
  int pixelFormat = ChoosePixelFormat(hdc, &pfd);
  XEASSERTNOTZERO(pixelFormat);
  int result = SetPixelFormat(hdc, pixelFormat, &pfd);
  // Create a dummy OpenGL context to initialize GLEW on.
  HGLRC context = wglCreateContext(hdc);
  XEASSERTNOTNULL(context);
  result = wglMakeCurrent(hdc, context);
  XEASSERTTRUE(result);
  result = glewInit();
  XEASSERT(result == GLEW_OK);
  result = wglDeleteContext(context);
  XEASSERTTRUE(result);
  context = NULL;
  // Now create the real OpenGL 4.4 context.
  const int createContextAttribs[] = {
    WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
    WGL_CONTEXT_MINOR_VERSION_ARB, 4,
    // GLEW does not support core profiles :(
    WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
#ifdef DEBUG
    WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
#endif
    0
  };
  context = wglCreateContextAttribsARB(hdc, NULL, createContextAttribs);
  XEASSERTNOTNULL(context);
  result = wglMakeCurrent(hdc, context);
  XEASSERTTRUE(result);
  result = glewInit();
  XEASSERT(result == GLEW_OK);
  if (GLEW_ARB_debug_output) {
    glDebugMessageCallback(&GLDebugMessage, NULL);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  }

  // Create the driver.
  // This runs in the worker thread and builds command lines to present
  // in the window.
  XEASSERTNULL(driver_);
  driver_ = new GLGraphicsDriver(memory_);

  // Initial vsync kick.
  DispatchInterruptCallback(0);
}

void GLGraphicsSystem::Pump() {
  if (swap_pending_) {
    swap_pending_ = false;

    // Swap window.
    // If we are set to vsync this will block.
    window_->Swap();

    DispatchInterruptCallback(0);
  } else {
    // If we have gone too long without an interrupt, fire one.
    if (xe_pal_now() - last_interrupt_time_ > 500 / 1000.0) {
      DispatchInterruptCallback(0);
    }
  }
}

void GLGraphicsSystem::Shutdown() {
  GraphicsSystem::Shutdown();
  
  if (vsync_timer_) {
    DeleteTimerQueueTimer(timer_queue_, vsync_timer_, NULL);
  }
  if (timer_queue_) {
    DeleteTimerQueueEx(timer_queue_, NULL);
  }

  delete window_;
  window_ = 0;
}
