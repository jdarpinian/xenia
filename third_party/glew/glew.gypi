# Copyright 2014 James Darpinian. All Rights Reserved.
{
  'targets': [
    {
      'target_name': 'glew',
      'type': '<(library)',

      'sources': [
        'glew.c',
      ],

      'defines': [
        'GLEW_STATIC',
      ],

      'include_dirs': [
        '.',
      ],

      'direct_dependent_settings': {
        'include_dirs': [
          '.',
        ],
        'defines': [
          'GLEW_STATIC',
        ],
      },
    }
  ]
}
