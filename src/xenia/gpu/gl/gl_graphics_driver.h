/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_GRAPHICS_DRIVER_H_
#define XENIA_GPU_GL_GL_GRAPHICS_DRIVER_H_

#include <xenia/core.h>

#include <xenia/gpu/graphics_driver.h>
#include <xenia/gpu/xenos/xenos.h>

#include <GL/glew.h>


namespace xe {
namespace gpu {
namespace gl {

class GLPixelShader;
class GLShaderCache;
class GLVertexShader;


class GLGraphicsDriver : public GraphicsDriver {
public:
  GLGraphicsDriver(Memory* memory);
  virtual ~GLGraphicsDriver();

  virtual void Initialize();

  virtual void InvalidateState(
      uint32_t mask);
  virtual void SetShader(
      xenos::XE_GPU_SHADER_TYPE type,
      uint32_t address,
      uint32_t start,
      uint32_t length);
  virtual void DrawIndexBuffer(
      xenos::XE_GPU_PRIMITIVE_TYPE prim_type,
      bool index_32bit, uint32_t index_count,
      uint32_t index_base, uint32_t index_size, uint32_t endianness);
  virtual void DrawIndexAuto(
      xenos::XE_GPU_PRIMITIVE_TYPE prim_type,
      uint32_t index_count);

private:
  int SetupDraw(xenos::XE_GPU_PRIMITIVE_TYPE prim_type);
  int UpdateState();
  int UpdateConstantBuffers();
  int BindShaders();
  int PrepareFetchers();
  int PrepareVertexFetcher(
      int fetch_slot, xenos::xe_gpu_vertex_fetch_t* fetch);
  int PrepareTextureFetcher(
      int fetch_slot, xenos::xe_gpu_texture_fetch_t* fetch);
  int PrepareIndexBuffer(
      bool index_32bit, uint32_t index_count,
      uint32_t index_base, uint32_t index_size, uint32_t endianness);

private:
  GLShaderCache*        shader_cache_;
  GLuint                pipeline_;
  GLuint                vertex_buffer_;
  GLuint                index_buffer_;


  struct {
    GLVertexShader*  vertex_shader;
    GLPixelShader*   pixel_shader;

    struct {
      GLuint floats;
      GLuint loops;
      GLuint bools;
    } constant_buffers;
  } state_;
};


}  // namespace gl
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_GRAPHICS_DRIVER_H_
