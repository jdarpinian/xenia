/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GL_GL_SHADER_H_
#define XENIA_GPU_GL_GL_SHADER_H_

#include <xenia/core.h>

#include <xenia/gpu/shader.h>
#include <xenia/gpu/xenos/xenos.h>

#include <GL/glew.h>


namespace xe {
namespace gpu {
namespace gl {

struct Output;

typedef struct {
  Output*       output;
  xenos::XE_GPU_SHADER_TYPE type;
} xe_gpu_translate_ctx_t;


class GLShader : public Shader {
public:
  virtual ~GLShader();

protected:
  GLShader(
      xenos::XE_GPU_SHADER_TYPE type,
      const uint8_t* src_ptr, size_t length,
      uint64_t hash);

  const char* translated_src() const { return translated_src_; }
  void set_translated_src(char* value);

  int TranslateExec(
      xe_gpu_translate_ctx_t& ctx, const xenos::instr_cf_exec_t& cf);

  GLuint Compile(const char* shader_source);

  virtual const char* ShaderTypeName() const = 0;
  virtual GLuint ShaderTypeEnum() const = 0;

protected:
  char* translated_src_;
};


class GLVertexShader : public GLShader {
public:
  GLVertexShader(
      const uint8_t* src_ptr, size_t length,
      uint64_t hash);
  virtual ~GLVertexShader();

  GLuint handle() const { return handle_; }
  GLuint vao() const { return vao_; };

  virtual const char* ShaderTypeName() const { return "vertex"; };
  virtual GLuint ShaderTypeEnum() const { return GL_VERTEX_SHADER; };

  int Prepare(xenos::xe_gpu_program_cntl_t* program_cntl);

private:
  const char* Translate(xenos::xe_gpu_program_cntl_t* program_cntl);

private:
  GLuint handle_;
  GLuint vao_;
};


class GLPixelShader : public GLShader {
public:
  GLPixelShader(
      const uint8_t* src_ptr, size_t length,
      uint64_t hash);
  virtual ~GLPixelShader();

  GLuint handle() const { return handle_; }
  virtual const char* ShaderTypeName() const { return "pixel"; };
  virtual GLuint ShaderTypeEnum() const { return GL_FRAGMENT_SHADER; };

  int Prepare(xenos::xe_gpu_program_cntl_t* program_cntl,
              GLVertexShader* input_shader);

private:
  const char* Translate(xenos::xe_gpu_program_cntl_t* program_cntl,
                        GLVertexShader* input_shader);

private:
  GLuint handle_;
};


}  // namespace d3d11
}  // namespace gpu
}  // namespace xe


#endif  // XENIA_GPU_GL_GL_SHADER_H_
