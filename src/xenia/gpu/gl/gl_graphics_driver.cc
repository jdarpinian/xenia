/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 James Darpinian. All rights reserved.                       *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/gpu/gl/gl_graphics_driver.h>

#include <xenia/gpu/gpu-private.h>
#include <xenia/gpu/gl/gl_shader.h>
#include <xenia/gpu/gl/gl_shader_cache.h>


using namespace xe;
using namespace xe::gpu;
using namespace xe::gpu::gl;
using namespace xe::gpu::xenos;


GLGraphicsDriver::GLGraphicsDriver(
    Memory* memory) :
    GraphicsDriver(memory) {
  shader_cache_ = new GLShaderCache();

  xe_zero_struct(&state_, sizeof(state_));

  glGenBuffers(3, &(state_.constant_buffers.floats));
  glGenBuffers(1, &vertex_buffer_);
  glGenBuffers(1, &index_buffer_);
  glGenProgramPipelines(1, &pipeline_);
  glBindProgramPipeline(pipeline_);
  // TODO: Cache pipelines instead of switching shaders in pipelines.
}

GLGraphicsDriver::~GLGraphicsDriver() {
  delete shader_cache_;
}

void GLGraphicsDriver::Initialize() {
}

void GLGraphicsDriver::InvalidateState(
    uint32_t mask) {
  if (mask == XE_GPU_INVALIDATE_MASK_ALL) {
    XELOGGPU("GL: (invalidate all)");
  }
  if (mask & XE_GPU_INVALIDATE_MASK_VERTEX_SHADER) {
    XELOGGPU("GL: invalidate vertex shader");
  }
  if (mask & XE_GPU_INVALIDATE_MASK_PIXEL_SHADER) {
    XELOGGPU("GL: invalidate pixel shader");
  }
}

void GLGraphicsDriver::SetShader(
    XE_GPU_SHADER_TYPE type,
    uint32_t address,
    uint32_t start,
    uint32_t length) {
  // Find or create shader in the cache.
  uint8_t* p = memory_->Translate(address);
  Shader* shader = shader_cache_->FindOrCreate(
      type, p, length);

  // Disassemble.
  const char* source = shader->disasm_src();
  if (!source) {
    source = "<failed to disassemble>";
  }
  XELOGGPU("GL: set shader %d at %0.8X (%db):\n%s",
           type, address, length, source);

  // Stash for later.
  switch (type) {
  case XE_GPU_SHADER_TYPE_VERTEX:
    state_.vertex_shader = (GLVertexShader*)shader;
    break;
  case XE_GPU_SHADER_TYPE_PIXEL:
    state_.pixel_shader = (GLPixelShader*)shader;
    break;
  }
}

int GLGraphicsDriver::SetupDraw(XE_GPU_PRIMITIVE_TYPE prim_type) {
  RegisterFile& rf = register_file_;

  // Misc state.
  if (UpdateState()) {
    return 1;
  }

  // Build constant buffers.
  if (UpdateConstantBuffers()) {
    return 1;
  }

  // Bind shaders.
  if (BindShaders()) {
    return 1;
  }
  
  // Setup all fetchers (vertices/textures).
  if (PrepareFetchers()) {
    return 1;
  }

  // All ready to draw (except index buffer)!
  return 0;
}

static GLenum gl_primitive_type(XE_GPU_PRIMITIVE_TYPE type) {
  switch (type) {
  case XE_GPU_PRIMITIVE_TYPE_POINT_LIST:
    return GL_POINTS;
  case XE_GPU_PRIMITIVE_TYPE_LINE_LIST:
    return GL_LINES;
  case XE_GPU_PRIMITIVE_TYPE_LINE_STRIP:
    return GL_LINE_STRIP;
  case XE_GPU_PRIMITIVE_TYPE_TRIANGLE_LIST:
    return GL_TRIANGLES;
  case XE_GPU_PRIMITIVE_TYPE_TRIANGLE_STRIP:
    return GL_TRIANGLE_STRIP;
  case XE_GPU_PRIMITIVE_TYPE_TRIANGLE_FAN:
    return GL_TRIANGLE_FAN;
  case XE_GPU_PRIMITIVE_TYPE_LINE_LOOP:
    return GL_LINE_LOOP;
  case XE_GPU_PRIMITIVE_TYPE_UNKNOWN_07:
  case XE_GPU_PRIMITIVE_TYPE_RECTANGLE_LIST:
  default:
    XELOGE("GL: unsupported primitive type %d", type);
    return GL_POINTS;
  }
}

void GLGraphicsDriver::DrawIndexBuffer(
    XE_GPU_PRIMITIVE_TYPE prim_type,
    bool index_32bit, uint32_t index_count,
    uint32_t index_base, uint32_t index_size, uint32_t endianness) {
  RegisterFile& rf = register_file_;

  XELOGGPU("GL: draw indexed %d (%d indicies) from %.8X",
           prim_type, index_count, index_base);

  // Setup shaders/etc.
  if (SetupDraw(prim_type)) {
    return;
  }

  // Setup index buffer.
  if (PrepareIndexBuffer(
      index_32bit, index_count, index_base, index_size, endianness)) {
    return;
  }

  // Issue draw.
  uint32_t index_size_bytes = index_32bit ? 4 : 2;
  uint32_t start_index = rf.values[XE_GPU_REG_VGT_INDX_OFFSET].u32;
  uint32_t base_vertex = 0;
  glDrawElements(gl_primitive_type(prim_type), index_count, index_32bit ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, (void*)(start_index * index_size_bytes));
}

void GLGraphicsDriver::DrawIndexAuto(
    XE_GPU_PRIMITIVE_TYPE prim_type,
    uint32_t index_count) {
  RegisterFile& rf = register_file_;

  XELOGGPU("GL: draw indexed %d (%d indicies)",
           prim_type, index_count);

  // Setup shaders/etc.
  if (SetupDraw(prim_type)) {
    return;
  }

  // Issue draw.
  uint32_t start_index = rf.values[XE_GPU_REG_VGT_INDX_OFFSET].u32;
  uint32_t base_vertex = 0;
  glDrawArrays(gl_primitive_type(prim_type), 0, index_count);
}

static void enable_gl_state(GLint state, bool enabled) {
  if (enabled) {
    glEnable(state);
  } else {
    glDisable(state);
  }
}

// Translates xenos stencil op enum values to the GL equivalents.
static GLint stencil_op_to_gl[] = {
  GL_KEEP,
  GL_ZERO,
  GL_REPLACE,
  GL_INCR,
  GL_DECR,
  GL_INVERT,
  GL_INCR_WRAP,
  GL_DECR_WRAP
};

int GLGraphicsDriver::UpdateState() {
  // https://chromium.googlesource.com/chromiumos/third_party/mesa/+/6173cc19c45d92ef0b7bc6aa008aa89bb29abbda/src/gallium/drivers/freedreno/freedreno_zsa.c
  // http://cgit.freedesktop.org/mesa/mesa/diff/?id=aac7f06ad843eaa696363e8e9c7781ca30cb4914
  RegisterFile& rf = register_file_;

  // RB_SURFACE_INFO
  // RB_DEPTH_INFO
  // General rasterizer state.
  // TODO: Optimize redundant state setting.
  uint32_t mode_control = rf.values[XE_GPU_REG_RB_MODECONTROL].u32;
  switch (mode_control & 0x3) {
  case 0:
    glDisable(GL_CULL_FACE);
    break;
  case 1:
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    break;
  case 2:
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    break;
  }
  glFrontFace((mode_control & 0x4) == 0 ? GL_CCW : GL_CW);

  // Depth-stencil state.
  uint32_t depth_control = rf.values[XE_GPU_REG_RB_DEPTHCONTROL].u32;
  // A2XX_RB_DEPTHCONTROL_BACKFACE_ENABLE
  // ?
  // A2XX_RB_DEPTHCONTROL_Z_ENABLE
  // TODO: Re-enable this after depth clear is working.
  enable_gl_state(GL_DEPTH_TEST, GL_FALSE); //(depth_control & 0x00000002) != 0);
  // A2XX_RB_DEPTHCONTROL_Z_WRITE_ENABLE
  glDepthMask((depth_control & 0x00000004) ? GL_TRUE : GL_FALSE);
  // A2XX_RB_DEPTHCONTROL_EARLY_Z_ENABLE
  // ?
  // A2XX_RB_DEPTHCONTROL_ZFUNC
  // 0 = never, 7 = always -- lines up exactly
  glDepthFunc(GL_NEVER + ((depth_control & 0x00000070) >> 4));
  // A2XX_RB_DEPTHCONTROL_STENCIL_ENABLE
  enable_gl_state(GL_STENCIL_TEST, (depth_control & 0x00000001) != 0);
  // A2XX_RB_DEPTHCONTROL_STENCILFUNC
  glStencilFuncSeparate(GL_FRONT, GL_NEVER + ((depth_control & 0x00000700) >> 8), 0, 0xFF);
  // 0 = keep, 7 = decr -- almost lines up
  glStencilOpSeparate(GL_FRONT,
      // A2XX_RB_DEPTHCONTROL_STENCILFAIL
      stencil_op_to_gl[((depth_control & 0x00003800) >> 11)],
      // A2XX_RB_DEPTHCONTROL_STENCILZFAIL
      stencil_op_to_gl[((depth_control & 0x000E0000) >> 17)],
      // A2XX_RB_DEPTHCONTROL_STENCILZPASS
      stencil_op_to_gl[((depth_control & 0x0001C000) >> 14)]);
  // A2XX_RB_DEPTHCONTROL_STENCILFUNC_BF
  glStencilFuncSeparate(GL_BACK, GL_NEVER + ((depth_control & 0x00700000) >> 20), 0, 0xFF);
  glStencilOpSeparate(GL_BACK,
      // A2XX_RB_DEPTHCONTROL_STENCILFAIL_BF
      stencil_op_to_gl[((depth_control & 0x03800000) >> 23)],
      // A2XX_RB_DEPTHCONTROL_STENCILZFAIL_BF
      stencil_op_to_gl[((depth_control & 0xE0000000) >> 29)],
      // A2XX_RB_DEPTHCONTROL_STENCILZPASS_BF
      stencil_op_to_gl[((depth_control & 0x1C000000) >> 26)]);

  // Blend state.
  // TODO
  //context_->OMSetBlendState(blend_state, blend_factor, sample_mask);

  // Scissoring.
  // TODO(benvanik): pull from scissor registers.
  
  // Viewport.
  // If we have resized the window we will want to change this.
  glViewport(0, 0, 1280, 720);
  glDepthRange(0, 1);

  return 0;
}

int GLGraphicsDriver::UpdateConstantBuffers() {
  RegisterFile& rf = register_file_;

  // TODO: Only upload state that's needed by the shader in use.
  // TODO: Don't upload unchanged state.
  glBindBuffer(GL_UNIFORM_BUFFER, state_.constant_buffers.floats);
  glBufferData(GL_UNIFORM_BUFFER, (512 * 4) * sizeof(float), &rf.values[XE_GPU_REG_SHADER_CONSTANT_000_X], GL_STREAM_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, state_.constant_buffers.loops);
  glBufferData(GL_UNIFORM_BUFFER, (32) * sizeof(int), &rf.values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00], GL_STREAM_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, state_.constant_buffers.bools);
  glBufferData(GL_UNIFORM_BUFFER, (8) * sizeof(int), &rf.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], GL_STREAM_DRAW);

  return 0;
}

int GLGraphicsDriver::BindShaders() {
  RegisterFile& rf = register_file_;
  xe_gpu_program_cntl_t program_cntl;
  program_cntl.dword_0 = rf.values[XE_GPU_REG_SQ_PROGRAM_CNTL].u32;

  // Vertex shader setup.
  GLVertexShader* vs = state_.vertex_shader;
  if (vs) {
    if (!vs->is_prepared()) {
      // Prepare for use.
      if (vs->Prepare(&program_cntl)) {
        XELOGGPU("GL: failed to prepare vertex shader");
        state_.vertex_shader = NULL;
        return 1;
      }
    }

    // Bind.
    // TODO: Create and cache pipeline objects for each used combination of vertex/pixel shaders
    // instead of reconstructing the pipeline every time.
    glUseProgramStages(pipeline_, GL_VERTEX_SHADER_BIT, vs->handle());
    glBindVertexArray(vs->vao());


    // Set constant buffers.
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, state_.constant_buffers.floats);
  } else {
    return 1;
  }

  // Pixel shader setup.
  GLPixelShader* ps = state_.pixel_shader;
  if (ps) {
    if (!ps->is_prepared()) {
      // Prepare for use.
      if (ps->Prepare(&program_cntl, vs)) {
        XELOGGPU("GL: failed to prepare pixel shader");
        state_.pixel_shader = NULL;
        return 1;
      }
    }

    // Bind.
    // TODO: Create and cache pipeline objects for each used combination of vertex/pixel shaders
    // instead of reconstructing the pipeline every time.
    glUseProgramStages(pipeline_, GL_FRAGMENT_SHADER_BIT, ps->handle());

    // Set constant buffers.
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, state_.constant_buffers.floats);

    // TODO: set texture sampling state
  } else {
    return 1;
  }

  return 0;
}

int GLGraphicsDriver::PrepareFetchers() {
  RegisterFile& rf = register_file_;
  for (int n = 0; n < 32; n++) {
    int r = XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + n * 6;
    xe_gpu_fetch_group_t* group = (xe_gpu_fetch_group_t*)&rf.values[r];
    if (group->type_0 == 0x2) {
      if (PrepareTextureFetcher(n, &group->texture_fetch)) {
        return 1;
      }
    } else {
      // TODO(benvanik): verify register numbering.
      if (group->type_0 == 0x3) {
        if (PrepareVertexFetcher(n * 3 + 0, &group->vertex_fetch_0)) {
          return 1;
        }
      }
      if (group->type_1 == 0x3) {
        if (PrepareVertexFetcher(n * 3 + 1, &group->vertex_fetch_1)) {
          return 1;
        }
      }
      if (group->type_2 == 0x3) {
        if (PrepareVertexFetcher(n * 3 + 2, &group->vertex_fetch_2)) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int GLGraphicsDriver::PrepareVertexFetcher(
    int fetch_slot, xe_gpu_vertex_fetch_t* fetch) {
  uint32_t address = (fetch->address << 2) + address_translation_;
  uint32_t size_dwords = fetch->size;

  int size_bytes = size_dwords * 4;
  uint32_t* src = (uint32_t*)memory_->Translate(address);
  // TODO: Avoid mallocing a temp array just for byte swapping.
  uint32_t* dest = (uint32_t*)malloc(size_bytes);
  for (uint32_t n = 0; n < size_dwords; n++) {
    // union {
    //   uint32_t i;
    //   float f;
    // } d = {XESWAP32(src[n])};
    // XELOGGPU("v%.3d %0.8X %g", n, d.i, d.f);
    dest[n] = XESWAP32(src[n]);
  }
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, size_bytes, dest, GL_STREAM_DRAW);
  free(dest);

  GLVertexShader* vs = state_.vertex_shader;
  if (!vs) {
    return 1;
  }
  const instr_fetch_vtx_t* vtx = vs->GetFetchVtxBySlot(fetch_slot);
  if (!vtx->must_be_one) {
    return 1;
  }
  // TODO(benvanik): always dword aligned?
  uint32_t stride = vtx->stride * 4;
  uint32_t offset = 0;
  uint32_t vb_slot = fetch_slot > 30 ? 95 - fetch_slot : fetch_slot;
  glBindVertexBuffer(vb_slot, vertex_buffer_, offset, stride);

  return 0;
}

int GLGraphicsDriver::PrepareTextureFetcher(
    int fetch_slot, xe_gpu_texture_fetch_t* fetch) {
  RegisterFile& rf = register_file_;

  // maybe << 2?
  uint32_t address = (fetch->address << 4) + address_translation_;
  return 0;
}

int GLGraphicsDriver::PrepareIndexBuffer(
    bool index_32bit, uint32_t index_count,
    uint32_t index_base, uint32_t index_size, uint32_t endianness) {
  XEASSERTALWAYS(); // TODO
  RegisterFile& rf = register_file_;

  uint32_t address = index_base + address_translation_;

  // All that's done so far:
  XEASSERT(endianness == 0x2);

  // TODO: avoid mallocing a temp buffer just for byte swapping.
  void* temp_buffer = malloc(index_size);
  if (index_32bit) {
    uint32_t* src = (uint32_t*)memory_->Translate(address);
    uint32_t* dest = (uint32_t*)temp_buffer;
    for (uint32_t n = 0; n < index_count; n++) {
      uint32_t d = { XESWAP32(src[n]) };
      //XELOGGPU("i%.4d %0.8X", n, d);
      dest[n] = d;
    }
  } else {
    uint16_t* src = (uint16_t*)memory_->Translate(address);
    uint16_t* dest = (uint16_t*)temp_buffer;
    for (uint32_t n = 0; n < index_count; n++) {
      uint16_t d = XESWAP16(src[n]);
      //XELOGGPU("i%.4d, %.4X", n, d);
      dest[n] = d;
    }
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, temp_buffer, GL_STREAM_DRAW);
  free(temp_buffer);

  return 0;
}
