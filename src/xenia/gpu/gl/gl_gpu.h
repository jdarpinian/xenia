/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_GPU_H_
#define XENIA_GPU_GL_GL_GPU_H_

#include <xenia/core.h>


XEDECLARECLASS1(xe, Emulator);
XEDECLARECLASS2(xe, gpu, GraphicsSystem);


namespace xe {
namespace gpu {
namespace gl {


GraphicsSystem* Create(Emulator* emulator);


}  // namespace gl
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_GPU_H_
