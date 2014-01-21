# Copyright 2013 Ben Vanik. All Rights Reserved.
{
  'sources': [
    'command_buffer.h',
    'gpu-private.h',
    'gpu.cc',
    'gpu.h',
    'graphics_driver.cc',
    'graphics_driver.h',
    'graphics_system.cc',
    'graphics_system.h',
    'ring_buffer_worker.cc',
    'ring_buffer_worker.h',
    'shader.cc',
    'shader.h',
    'shader_cache.cc',
    'shader_cache.h',
  ],

  'includes': [
    'gl/sources.gypi',
    'nop/sources.gypi',
    'xenos/sources.gypi',
  ],

  'conditions': [
    ['OS == "win"', {
      'includes': [
        'd3d11/sources.gypi',
      ],
    }],
  ],
}
