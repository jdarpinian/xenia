/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_GRAPHICS_SYSTEM_H_
#define XENIA_GPU_GL_GL_GRAPHICS_SYSTEM_H_

#include <xenia/core.h>

#include <xenia/gpu/graphics_system.h>

#include <GL/glew.h>
#include <GL/wglew.h>


namespace xe {
namespace gpu {
namespace gl {

class GLWindow;


GraphicsSystem* Create(Emulator* emulator);


class GLGraphicsSystem : public GraphicsSystem {
public:
  GLGraphicsSystem(Emulator* emulator);
  virtual ~GLGraphicsSystem();

  virtual void Shutdown();

protected:
  virtual void Initialize();
  virtual void Pump();

private:
  GLWindow*    window_;

  HANDLE          timer_queue_;
  HANDLE          vsync_timer_;
};


}  // namespace gl
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_GRAPHICS_SYSTEM_H_
