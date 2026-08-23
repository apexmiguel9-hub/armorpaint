#pragma once

#include <GLES3/gl32.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdbool.h>
#include <stdint.h>

#define GPU_MAX_TEXTURES             16
#define GPU_MAX_VERTEX_ELEMENTS      16

typedef struct {
	GLuint texture;
	GLuint fbo;
	GLuint depth_renderbuffer;
	GLuint width;
	GLuint height;
	GLenum gl_format;
	GLenum gl_type;
	GLenum gl_internal_format;
	bool   is_render_target;
} gpu_texture_impl_t;

typedef struct {
	char *source;
	int   length;
} gpu_shader_impl_t;

typedef struct {
	GLuint program;
	GLuint vao;
} gpu_pipeline_impl_t;

typedef struct {
	GLuint vbo;
} gpu_buffer_impl_t;

typedef struct {
	GLuint vao;
} gpu_acceleration_structure_impl_t;

void gpu_lazy_flush();

