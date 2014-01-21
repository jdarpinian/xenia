/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_SHADER_CACHE_H_
#define XENIA_GPU_GL_GL_SHADER_CACHE_H_

#include <xenia/core.h>

#include <xenia/gpu/shader_cache.h>

#include <GL/glew.h>


namespace xe {
namespace gpu {
namespace gl {


class GLShaderCache : public ShaderCache {
public:
  GLShaderCache();
  virtual ~GLShaderCache();

protected:
  virtual Shader* CreateCore(
      xenos::XE_GPU_SHADER_TYPE type,
      const uint8_t* src_ptr, size_t length,
      uint64_t hash);

protected:
};


}  // namespace gl
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_SHADER_CACHE_H_
