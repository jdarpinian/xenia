/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/gpu/gl/gl_shader_cache.h>

#include <xenia/gpu/gl/gl_shader.h>


using namespace xe;
using namespace xe::gpu;
using namespace xe::gpu::gl;
using namespace xe::gpu::xenos;


GLShaderCache::GLShaderCache() {
}

GLShaderCache::~GLShaderCache() {
}

Shader* GLShaderCache::CreateCore(
    xenos::XE_GPU_SHADER_TYPE type,
    const uint8_t* src_ptr, size_t length,
    uint64_t hash) {
  switch (type) {
  case XE_GPU_SHADER_TYPE_VERTEX:
    return new GLVertexShader(
        src_ptr, length, hash);
  case XE_GPU_SHADER_TYPE_PIXEL:
    return new GLPixelShader(
        src_ptr, length, hash);
  default:
    XEASSERTALWAYS();
    return NULL;
  }
}