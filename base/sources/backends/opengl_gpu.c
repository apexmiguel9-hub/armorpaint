#include "opengl_gpu.h"
#include <iron_gpu.h>
#include <iron_log.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GLCHECK() do { GLenum err = glGetError(); if (err != GL_NO_ERROR) iron_log("GL error 0x%x at %s:%d", err, __FILE__, __LINE__); } while(0)

static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLContext egl_context = EGL_NO_CONTEXT;
static EGLSurface egl_surface = EGL_NO_SURFACE;

static gpu_texture_t *current_textures[GPU_MAX_TEXTURES] = {0};
static gpu_texture_t *current_render_targets[8] = {0};
static uint32_t current_render_targets_count = 0;
static gpu_texture_t *current_depth_buffer = NULL;
static gpu_pipeline_t *current_pipeline = NULL;
static uint32_t constant_buffer_index = 0;
static gpu_texture_t framebuffers[GPU_FRAMEBUFFER_COUNT];
static gpu_texture_t framebuffer_depth;
static uint32_t framebuffer_index = 0;

static int window_width = 0;
static int window_height = 0;

static bool gpu_in_use = false;
static bool gpu_raytrace_multi = false;

static uint32_t gpu_vertex_data_size(gpu_vertex_data_t data) {
	switch (data) {
	case GPU_VERTEX_DATA_F32_1X: return 4;
	case GPU_VERTEX_DATA_F32_2X: return 8;
	case GPU_VERTEX_DATA_F32_3X: return 12;
	case GPU_VERTEX_DATA_F32_4X: return 16;
	case GPU_VERTEX_DATA_I16_2X_NORM: return 4;
	case GPU_VERTEX_DATA_I16_4X_NORM: return 8;
	default: return 16;
	}
}

static uint32_t gpu_vertex_struct_size(gpu_vertex_structure_t *s) {
	uint32_t size = 0;
	for (uint32_t i = 0; i < s->size; ++i) {
		size += gpu_vertex_data_size(s->elements[i].data);
	}
	return size;
}

static void gl_format(gpu_texture_format_t format, GLint *internal, GLenum *fmt, GLenum *type) {
	switch (format) {
	case GPU_TEXTURE_FORMAT_RGBA32:
		*internal = GL_RGBA8;
		*fmt = GL_RGBA;
		*type = GL_UNSIGNED_BYTE;
		break;
	case GPU_TEXTURE_FORMAT_RGBA64:
		*internal = GL_RGBA16F;
		*fmt = GL_RGBA;
		*type = GL_HALF_FLOAT;
		break;
	case GPU_TEXTURE_FORMAT_RGBA128:
		*internal = GL_RGBA32F;
		*fmt = GL_RGBA;
		*type = GL_FLOAT;
		break;
	case GPU_TEXTURE_FORMAT_R8:
		*internal = GL_R8;
		*fmt = GL_RED;
		*type = GL_UNSIGNED_BYTE;
		break;
	case GPU_TEXTURE_FORMAT_R16:
		*internal = GL_R16F;
		*fmt = GL_RED;
		*type = GL_HALF_FLOAT;
		break;
	case GPU_TEXTURE_FORMAT_R32:
		*internal = GL_R32F;
		*fmt = GL_RED;
		*type = GL_FLOAT;
		break;
	case GPU_TEXTURE_FORMAT_D32:
		*internal = GL_DEPTH_COMPONENT32F;
		*fmt = GL_DEPTH_COMPONENT;
		*type = GL_FLOAT;
		break;
	case GPU_TEXTURE_FORMAT_RGBA32_BC7:
		*internal = GL_COMPRESSED_RGBA_BPTC_UNORM;
		*fmt = GL_RGBA;
		*type = GL_UNSIGNED_BYTE;
		break;
	default:
		*internal = GL_RGBA8;
		*fmt = GL_RGBA;
		*type = GL_UNSIGNED_BYTE;
		break;
	}
}

static bool egl_init() {
	if (egl_display != EGL_NO_DISPLAY) return true;

	egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (egl_display == EGL_NO_DISPLAY) {
		iron_log("EGL: eglGetDisplay failed");
		return false;
	}

	EGLint major, minor;
	if (!eglInitialize(egl_display, &major, &minor)) {
		iron_log("EGL: eglInitialize failed");
		return false;
	}
	iron_log("EGL: version %d.%d", major, minor);

	const EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 24,
		EGL_STENCIL_SIZE, 8,
		EGL_SAMPLE_BUFFERS, 0,
		EGL_NONE
	};
	EGLConfig config;
	EGLint num_configs;
	if (!eglChooseConfig(egl_display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
		iron_log("EGL: eglChooseConfig failed");
		return false;
	}

	const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 3,
		EGL_NONE
	};
	egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, context_attribs);
	if (egl_context == EGL_NO_CONTEXT) {
		iron_log("EGL: eglCreateContext failed");
		return false;
	}

	iron_log("EGL: context created (GLES 3.x)");
	return true;
}

int iron_window_width(void) { return window_width; }
int iron_window_height(void) { return window_height; }

void *iron_android_get_native_window(void) { return NULL; }
int iron_android_get_native_window_width(void) { return window_width; }
int iron_android_get_native_window_height(void) { return window_height; }

void gpu_lazy_flush() {}

bool gpu_cleanup_pending(void) { return false; }
void gpu_cleanup() {}

bool gpu_bc7_supported(int width, int height, gpu_texture_format_t format) { return false; }

#ifdef WITH_BC7
void *gpu_bc7_compress(void *data, int width, int height) { return NULL; }
#endif

bool gpu_raytrace_supported(void) { return false; }
void gpu_raytrace_pipeline_init(gpu_raytrace_pipeline_t *pipeline, void *shader, int shader_size, gpu_buffer_t *constant_buffer) {}
void gpu_raytrace_pipeline_destroy(gpu_raytrace_pipeline_t *pipeline) {}
void gpu_raytrace_acceleration_structure_init(gpu_acceleration_structure_t *accel) {}
void gpu_raytrace_acceleration_structure_add(gpu_acceleration_structure_t *accel, gpu_buffer_t *vb, gpu_buffer_t *ib, mat4_t transform) {}
void gpu_raytrace_acceleration_structure_build(gpu_acceleration_structure_t *accel, gpu_buffer_t *vb_full, gpu_buffer_t *ib_full) {}
void gpu_raytrace_acceleration_structure_destroy(gpu_acceleration_structure_t *accel) {}
void gpu_raytrace_set_textures(gpu_texture_t *tex0, gpu_texture_t *tex1, gpu_texture_t *tex2, gpu_texture_t *texenv, gpu_texture_t *texsobol, gpu_texture_t *texscramble, gpu_texture_t *texrank) {}
void gpu_raytrace_set_acceleration_structure(gpu_acceleration_structure_t *accel) {}
void gpu_raytrace_set_pipeline(gpu_raytrace_pipeline_t *pipeline) {}
void gpu_raytrace_set_target(gpu_texture_t *output) {}
void gpu_raytrace_dispatch_rays() {}
void _gpu_raytrace_init(buffer_t *shader) {}
void _gpu_raytrace_as_init() {}
void _gpu_raytrace_as_add(gpu_buffer_t *vb, gpu_buffer_t *ib, mat4_t transform) {}
void _gpu_raytrace_as_build(gpu_buffer_t *vb_full, gpu_buffer_t *ib_full) {}
void _gpu_raytrace_dispatch_rays(gpu_texture_t *render_target, buffer_t *buffer) {}

char *gpu_device_name() { return (char*)"OpenGL ES 3.x"; }


static void apply_pipeline_state(gpu_pipeline_t *pipeline) {
	switch (pipeline->cull_mode) {
	case GPU_CULL_MODE_CLOCKWISE:
		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
		break;
	case GPU_CULL_MODE_COUNTER_CLOCKWISE:
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		break;
	case GPU_CULL_MODE_NONE:
		glDisable(GL_CULL_FACE);
		break;
	}

	if (pipeline->depth_write) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}

	switch (pipeline->depth_mode) {
	case GPU_COMPARE_MODE_ALWAYS: glDepthFunc(GL_ALWAYS); break;
	case GPU_COMPARE_MODE_NEVER:  glDepthFunc(GL_NEVER); break;
	case GPU_COMPARE_MODE_LESS:   glDepthFunc(GL_LESS); break;
	case GPU_COMPARE_MODE_EQUAL:  glDepthFunc(GL_EQUAL); break;
	}

	GLenum src, dst, asrc, adst;
	switch (pipeline->blend_source) {
	case GPU_BLEND_ONE: src = GL_ONE; break;
	case GPU_BLEND_ZERO: src = GL_ZERO; break;
	case GPU_BLEND_SOURCE_ALPHA: src = GL_SRC_ALPHA; break;
	case GPU_BLEND_DEST_ALPHA: src = GL_DST_ALPHA; break;
	case GPU_BLEND_INV_SOURCE_ALPHA: src = GL_ONE_MINUS_SRC_ALPHA; break;
	case GPU_BLEND_INV_DEST_ALPHA: src = GL_ONE_MINUS_DST_ALPHA; break;
	default: src = GL_ONE; break;
	}
	switch (pipeline->blend_destination) {
	case GPU_BLEND_ONE: dst = GL_ONE; break;
	case GPU_BLEND_ZERO: dst = GL_ZERO; break;
	case GPU_BLEND_SOURCE_ALPHA: dst = GL_SRC_ALPHA; break;
	case GPU_BLEND_DEST_ALPHA: dst = GL_DST_ALPHA; break;
	case GPU_BLEND_INV_SOURCE_ALPHA: dst = GL_ONE_MINUS_SRC_ALPHA; break;
	case GPU_BLEND_INV_DEST_ALPHA: dst = GL_ONE_MINUS_DST_ALPHA; break;
	default: dst = GL_ZERO; break;
	}
	switch (pipeline->alpha_blend_source) {
	case GPU_BLEND_ONE: asrc = GL_ONE; break;
	case GPU_BLEND_ZERO: asrc = GL_ZERO; break;
	case GPU_BLEND_INV_SOURCE_ALPHA: asrc = GL_ONE_MINUS_SRC_ALPHA; break;
	default: asrc = GL_SRC_ALPHA; break;
	}
	switch (pipeline->alpha_blend_destination) {
	case GPU_BLEND_ONE: adst = GL_ONE; break;
	case GPU_BLEND_ZERO: adst = GL_ZERO; break;
	case GPU_BLEND_INV_SOURCE_ALPHA: adst = GL_ONE_MINUS_SRC_ALPHA; break;
	default: adst = GL_ONE_MINUS_SRC_ALPHA; break;
	}

	if (src == GL_ONE && dst == GL_ZERO && asrc == GL_ONE && adst == GL_ZERO) {
		glDisable(GL_BLEND);
	} else {
		glEnable(GL_BLEND);
		glBlendFuncSeparate(src, dst, asrc, adst);
	}
	GLCHECK();
}

void gpu_set_pipeline_internal(gpu_pipeline_t *pipeline) {
	current_pipeline = pipeline;
	if (pipeline && pipeline->impl.program) {
		glUseProgram(pipeline->impl.program);
		apply_pipeline_state(pipeline);

		if (pipeline->impl.vao == 0) {
			glGenVertexArrays(1, &pipeline->impl.vao);
		}
		glBindVertexArray(pipeline->impl.vao);
	}
	GLCHECK();
}

void gpu_set_pipeline(gpu_pipeline_t *pipeline) {
	for (int i = 0; i < GPU_MAX_TEXTURES; ++i) current_textures[i] = NULL;
	if (pipeline && pipeline->impl.program == 0) return;
	gpu_set_pipeline_internal(pipeline);
}

static void bind_sampler_units(gpu_pipeline_t *pipeline) {
	static gpu_pipeline_t *bound_pipeline = NULL;
	if (bound_pipeline == pipeline) return;
	bound_pipeline = pipeline;

	if (!pipeline || !pipeline->impl.program) return;

	glUseProgram(pipeline->impl.program);

	GLint count = 0;
	glGetProgramiv(pipeline->impl.program, GL_ACTIVE_UNIFORMS, &count);
	char name[256];
	for (GLint i = 0; i < count; ++i) {
		GLsizei length = 0;
		GLenum type = 0;
		GLint size = 0;
		glGetActiveUniform(pipeline->impl.program, i, sizeof(name), &length, &size, &type, name);
		if (type == GL_SAMPLER_2D || type == GL_SAMPLER_CUBE || type == GL_SAMPLER_3D ||
		    type == GL_SAMPLER_2D_ARRAY || type == GL_IMAGE_2D || type == GL_UNSIGNED_INT_SAMPLER_2D) {
			const char *pfx = "SPIRV_Cross_Combined";
			int unit = 0;
			if (strncmp(name, pfx, strlen(pfx)) == 0 && name[strlen(pfx)] != '\0') {
				unit = atoi(name + strlen(pfx));
			}
			if (unit >= 0 && unit < GPU_MAX_TEXTURES) {
				GLint loc = glGetUniformLocation(pipeline->impl.program, name);
				if (loc >= 0) glUniform1i(loc, unit);
			}
		}
	}

	GLint block_count = 0;
	glGetProgramiv(pipeline->impl.program, GL_ACTIVE_UNIFORM_BLOCKS, &block_count);
	for (GLint b = 0; b < block_count; ++b) {
		char bname[128];
		bname[0] = 0;
		glGetActiveUniformBlockName(pipeline->impl.program, b, sizeof(bname), NULL, bname);
		glUniformBlockBinding(pipeline->impl.program, b, 0);
	}
	GLCHECK();
}

static GLuint white_fallback = 0;
static GLuint get_white_fallback() {
	if (white_fallback == 0) {
		unsigned char px[4] = {255, 255, 255, 255};
		glGenTextures(1, &white_fallback);
		glBindTexture(GL_TEXTURE_2D, white_fallback);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
	return white_fallback;
}

static void fill_unbound_textures() {
	GLuint fb = get_white_fallback();
	for (int i = 0; i < GPU_MAX_TEXTURES; ++i) {
		if (current_textures[i] == NULL || current_textures[i]->impl.texture == 0) {
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, fb);
		}
	}
}

void gpu_draw_internal() {
	if (!current_pipeline) return;

	bind_sampler_units(current_pipeline);
	fill_unbound_textures();

	GLint count = current_pipeline->impl.vao ? 0 : 0;
	if (current_pipeline->impl.vao) {
		glBindVertexArray(current_pipeline->impl.vao);
	}

	GLint idx_count = _current_index_count;
	if (idx_count > 0) {
		glDrawElements(GL_TRIANGLES, idx_count, GL_UNSIGNED_INT, 0);
	}
	GLCHECK();
}

static GLuint _current_index_buffer = 0;
static uint32_t _current_index_count = 0;

void gpu_set_index_buffer(gpu_buffer_t *buffer) {
	_current_index_buffer = buffer ? buffer->impl.vbo : 0;
	_current_index_count = buffer ? buffer->count : 0;
	if (buffer) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer->impl.vbo);
	} else {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	GLCHECK();
}

void *gpu_index_buffer_lock(gpu_buffer_t *buffer) {
	if (!buffer || buffer->data) return NULL;
	buffer->data = malloc(buffer->count * 4);
	return buffer->data;
}

void gpu_index_buffer_unlock(gpu_buffer_t *buffer) {
	if (!buffer || !buffer->data) return;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer->impl.vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, buffer->count * 4, buffer->data, GL_DYNAMIC_DRAW);
	free(buffer->data);
	buffer->data = NULL;
	GLCHECK();
}

void gpu_constant_buffer_init(gpu_buffer_t *buffer, uint32_t size) {
	buffer->count = size;
	buffer->data = malloc(size);
	memset(buffer->data, 0, size);
}

void *gpu_constant_buffer_lock(gpu_buffer_t *buffer, uint32_t start, uint32_t count) {
	if (!buffer->data) buffer->data = malloc(buffer->count);
	return buffer->data + start;
}

void gpu_constant_buffer_unlock(gpu_buffer_t *buffer) {}

void gpu_set_constant_buffer(gpu_buffer_t *buffer, uint32_t offset, size_t size) {
	if (!buffer || !buffer->data) return;
	if (offset + size > buffer->count) return;

	static GLuint ubo = 0;
	if (ubo == 0) glGenBuffers(1, &ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	glBufferData(GL_UNIFORM_BUFFER, size, buffer->data + offset, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

	if (current_pipeline && current_pipeline->impl.program) {
		GLint block_count = 0;
		glGetProgramiv(current_pipeline->impl.program, GL_ACTIVE_UNIFORM_BLOCKS, &block_count);
		for (GLint b = 0; b < block_count; ++b) {
			char bname[128]; bname[0] = 0;
			glGetActiveUniformBlockName(current_pipeline->impl.program, b, sizeof(bname), NULL, bname);
			glUniformBlockBinding(current_pipeline->impl.program, b, 0);
		}
	}
	GLCHECK();
}

void gpu_set_texture(uint32_t unit, gpu_texture_t *texture) {
	if (unit >= GPU_MAX_TEXTURES) return;
	current_textures[unit] = texture;
	if (!texture) return;

	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, texture->impl.texture);
	GLCHECK();
}

void gpu_use_linear_sampling(bool b) {
	for (int i = 0; i < GPU_MAX_TEXTURES; ++i) {
		if (current_textures[i] && current_textures[i]->impl.texture) {
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, current_textures[i]->impl.texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, b ? GL_LINEAR : GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, b ? GL_LINEAR : GL_NEAREST);
		}
	}
	GLCHECK();
}

static GLuint create_shader(GLenum type, const char *source) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);

	GLint compiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		GLint log_len = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0) {
			char *log = malloc(log_len + 1);
			glGetShaderInfoLog(shader, log_len + 1, NULL, log);
			iron_log("GL: shader compile failed: %s", log);
			free(log);
		}
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}

void gpu_shader_init(gpu_shader_t *shader, const void *source, size_t length, gpu_shader_type_t type) {
	shader->impl.source = malloc(length + 1);
	memcpy(shader->impl.source, source, length);
	shader->impl.source[length] = '\0';
	shader->impl.length = length;
	shader->impl.type = type;
}

void gpu_shader_destroy(gpu_shader_t *shader) {
	if (shader->impl.source) {
		free(shader->impl.source);
		shader->impl.source = NULL;
	}
	shader->impl.length = 0;
}

void gpu_pipeline_compile(gpu_pipeline_t *pipeline) {
	if (!pipeline->vertex_shader || !pipeline->fragment_shader) return;
	if (pipeline->vertex_shader->impl.length == 0 || pipeline->fragment_shader->impl.length == 0) return;

	GLuint vs = create_shader(GL_VERTEX_SHADER, pipeline->vertex_shader->impl.source);
	GLuint fs = create_shader(GL_FRAGMENT_SHADER, pipeline->fragment_shader->impl.source);
	if (!vs || !fs) {
		if (vs) glDeleteShader(vs);
		if (fs) glDeleteShader(fs);
		return;
	}

	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);

	GLint linked = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked) {
		GLint log_len = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0) {
			char *log = malloc(log_len + 1);
			glGetProgramInfoLog(program, log_len + 1, NULL, log);
			iron_log("GL: program link failed: %s", log);
			free(log);
		}
		glDeleteProgram(program);
		glDeleteShader(vs);
		glDeleteShader(fs);
		return;
	}

	glDeleteShader(vs);
	glDeleteShader(fs);

	if (pipeline->impl.program) glDeleteProgram(pipeline->impl.program);
	pipeline->impl.program = program;
	pipeline->impl.vao = 0;
	GLCHECK();
}

void gpu_pipeline_destroy(gpu_pipeline_t *pipeline) {
	if (pipeline->impl.program) {
		glDeleteProgram(pipeline->impl.program);
		pipeline->impl.program = 0;
	}
	if (pipeline->impl.vao) {
		glDeleteVertexArrays(1, &pipeline->impl.vao);
		pipeline->impl.vao = 0;
	}
}

void gpu_pipeline_destroy_internal(gpu_pipeline_t *pipeline) {
	gpu_pipeline_destroy(pipeline);
}

void gpu_pipeline_init(gpu_pipeline_t *pipeline) {
	pipeline->impl.program = 0;
	pipeline->impl.vao = 0;
}

void gpu_vertex_structure_add(gpu_vertex_structure_t *structure, const char *name, gpu_vertex_data_t data) {
	if (structure->size < GPU_MAX_VERTEX_ELEMENTS) {
		structure->elements[structure->size].name = name;
		structure->elements[structure->size].data = data;
		structure->size++;
	}
}

void gpu_texture_init_from_bytes(gpu_texture_t *texture, void *data, uint32_t width, uint32_t height, gpu_texture_format_t format) {
	texture->width = width;
	texture->height = height;
	texture->format = format;

	glGenTextures(1, &texture->impl.texture);
	glBindTexture(GL_TEXTURE_2D, texture->impl.texture);

	GLint internal; GLenum fmt, type;
	gl_format(format, &internal, &fmt, &type);

	glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, type, data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	texture->impl.fbo = 0;
	texture->impl.depth_renderbuffer = 0;
	texture->impl.is_render_target = false;
	glBindTexture(GL_TEXTURE_2D, 0);
	GLCHECK();
}

void gpu_texture_destroy(gpu_texture_t *texture) {
	if (texture->impl.texture) {
		glDeleteTextures(1, &texture->impl.texture);
		texture->impl.texture = 0;
	}
	if (texture->impl.fbo) {
		glDeleteFramebuffers(1, &texture->impl.fbo);
		texture->impl.fbo = 0;
	}
	if (texture->impl.depth_renderbuffer) {
		glDeleteRenderbuffers(1, &texture->impl.depth_renderbuffer);
		texture->impl.depth_renderbuffer = 0;
	}
}

void gpu_texture_destroy_internal(gpu_texture_t *texture) {
	gpu_texture_destroy(texture);
}

void gpu_render_target_init2(gpu_texture_t *render_target, uint32_t width, uint32_t height, gpu_texture_format_t format, int framebuffer_index) {
	render_target->width = width;
	render_target->height = height;
	render_target->format = format;

	GLint internal; GLenum fmt, type;
	gl_format(format, &internal, &fmt, &type);

	glGenTextures(1, &render_target->impl.texture);
	glBindTexture(GL_TEXTURE_2D, render_target->impl.texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, type, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	if (format == GPU_TEXTURE_FORMAT_D32) {
		render_target->impl.depth_renderbuffer = 0;
	} else {
		glGenRenderbuffers(1, &render_target->impl.depth_renderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, render_target->impl.depth_renderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);

		glGenFramebuffers(1, &render_target->impl.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, render_target->impl.fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_target->impl.texture, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_target->impl.depth_renderbuffer);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			iron_log("GL: render target fbo incomplete 0x%x", status);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	render_target->impl.is_render_target = true;
	render_target->impl.width = width;
	render_target->impl.height = height;
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	GLCHECK();
}

void gpu_render_target_init(gpu_texture_t *target, uint32_t width, uint32_t height, gpu_texture_format_t format) {
	gpu_render_target_init2(target, width, height, format, 0);
}

static GLuint get_composite_fbo(int color_count, GLuint *color_textures, GLuint depth_texture) {
	static GLuint cache_fbo[16] = {0};
	static GLuint cache_colors[16][4];
	static GLuint cache_depth[16];
	static int cache_count[16];
	static int cache_size = 0;

	for (int i = 0; i < cache_size; ++i) {
		if (cache_count[i] == color_count && cache_depth[i] == depth_texture) {
			bool match = true;
			for (int c = 0; c < color_count; ++c) {
				if (cache_colors[i][c] != color_textures[c]) { match = false; break; }
			}
			if (match) return cache_fbo[i];
		}
	}

	if (cache_size >= 16) return 0;
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	for (int c = 0; c < color_count; ++c) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + c, GL_TEXTURE_2D, color_textures[c], 0);
	}
	if (depth_texture) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0);
	}

	if (color_count > 0) {
		GLenum draw_bufs[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
		glDrawBuffers(color_count, draw_bufs);
	} else {
		glDrawBuffers(0, NULL);
	}

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		iron_log("GL: composite fbo incomplete 0x%x", status);
	}

	cache_fbo[cache_size] = fbo;
	cache_count[cache_size] = color_count;
	cache_depth[cache_size] = depth_texture;
	for (int c = 0; c < color_count; ++c) cache_colors[cache_size][c] = color_textures[c];
	cache_size++;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return fbo;
}

void gpu_begin_internal(gpu_clear_t flags, unsigned color, float depth) {
	if (egl_display == EGL_NO_DISPLAY) egl_init();

	if (current_render_targets_count > 0 && current_render_targets[0] && current_render_targets[0]->impl.is_render_target) {
		GLuint colors[4]; int color_count = 0; GLuint depth_tex = 0;
		for (int i = 0; i < current_render_targets_count && i < 4 && current_render_targets[i]; ++i) {
			if (current_render_targets[i]->format == GPU_TEXTURE_FORMAT_D32) {
				depth_tex = current_render_targets[i]->impl.texture;
			} else {
				colors[color_count++] = current_render_targets[i]->impl.texture;
			}
		}
		if (current_depth_buffer) depth_tex = current_depth_buffer->impl.texture;

		GLuint fbo = get_composite_fbo(color_count, colors, depth_tex);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	} else {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	if (flags & GPU_CLEAR_COLOR) {
		float r = ((color & 0x00ff0000) >> 16) / 255.0f;
		float g = ((color & 0x0000ff00) >> 8) / 255.0f;
		float b = ((color & 0x000000ff)) / 255.0f;
		float a = ((color & 0xff000000) >> 24) / 255.0f;
		glClearColor(r, g, b, a);
		glClear(GL_COLOR_BUFFER_BIT);
	}
	if (flags & GPU_CLEAR_DEPTH) {
		glClearDepthf(depth);
		glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}
	GLCHECK();
}

void gpu_end_internal() {}

void gpu_execute_and_wait() { glFinish(); }

void gpu_begin(gpu_texture_t **targets, int count, gpu_texture_t *depth_buffer, gpu_clear_t flags, unsigned color, float depth) {
	current_render_targets_count = count;
	for (int i = 0; i < count && i < 8; ++i) current_render_targets[i] = targets ? targets[i] : NULL;
	current_depth_buffer = depth_buffer;
	gpu_begin_internal(flags, color, depth);
}

void gpu_end() {}

void gpu_present_internal() {
	if (egl_display == EGL_NO_DISPLAY || egl_surface == EGL_NO_SURFACE) return;
	eglSwapBuffers(egl_display, egl_surface);
}

void gpu_present() {
	gpu_lazy_flush();
	gpu_present_internal();
}

void gpu_resize_internal(int width, int height) {
	window_width = width;
	window_height = height;
}

void gpu_resize(int width, int height) {
	gpu_resize_internal(width, height);
	for (int i = 0; i < GPU_FRAMEBUFFER_COUNT; ++i) {
		if (framebuffers[i].impl.texture) {
			glDeleteTextures(1, &framebuffers[i].impl.texture);
			framebuffers[i].impl.texture = 0;
		}
		if (framebuffers[i].impl.fbo) {
			glDeleteFramebuffers(1, &framebuffers[i].impl.fbo);
			framebuffers[i].impl.fbo = 0;
		}
	}
	if (framebuffer_depth.impl.texture) {
		glDeleteTextures(1, &framebuffer_depth.impl.texture);
		framebuffer_depth.impl.texture = 0;
	}
}

void gpu_viewport(int x, int y, int width, int height) { glViewport(x, y, width, height); }
void gpu_scissor(int x, int y, int width, int height) { glEnable(GL_SCISSOR_TEST); glScissor(x, y, width, height); }
void gpu_disable_scissor() { glDisable(GL_SCISSOR_TEST); }

void gpu_vertex_buffer_init(gpu_buffer_t *buffer, uint32_t count, gpu_vertex_structure_t *structure) {
	buffer->count = count;
	buffer->stride = gpu_vertex_struct_size(structure);
	buffer->data = NULL;
	glGenBuffers(1, &buffer->impl.vbo);
}

void *gpu_vertex_buffer_lock(gpu_buffer_t *buffer) {
	if (!buffer->data) buffer->data = malloc(buffer->stride * buffer->count);
	return buffer->data;
}

void gpu_vertex_buffer_unlock(gpu_buffer_t *buffer) {
	if (!buffer || !buffer->data) return;
	glBindBuffer(GL_ARRAY_BUFFER, buffer->impl.vbo);
	glBufferData(GL_ARRAY_BUFFER, buffer->stride * buffer->count, buffer->data, GL_DYNAMIC_DRAW);
	free(buffer->data);
	buffer->data = NULL;
	GLCHECK();
}

void gpu_buffer_destroy(gpu_buffer_t *buffer) {
	if (buffer->impl.vbo) glDeleteBuffers(1, &buffer->impl.vbo);
	if (buffer->data) free(buffer->data);
	buffer->data = NULL;
}

void gpu_buffer_destroy_internal(gpu_buffer_t *buffer) { gpu_buffer_destroy(buffer); }

void gpu_set_vertex_buffer(gpu_buffer_t *buffer) {
	if (!buffer) return;
	glBindBuffer(GL_ARRAY_BUFFER, buffer->impl.vbo);
	GLCHECK();
}

uint32_t gpu_vertex_data_size(gpu_vertex_data_t data) {
	switch (data) {
	case GPU_VERTEX_DATA_F32_1X: return 4;
	case GPU_VERTEX_DATA_F32_2X: return 8;
	case GPU_VERTEX_DATA_F32_3X: return 12;
	case GPU_VERTEX_DATA_F32_4X: return 16;
	case GPU_VERTEX_DATA_I16_2X_NORM: return 4;
	case GPU_VERTEX_DATA_I16_4X_NORM: return 8;
	default: return 16;
	}
}

void gpu_draw() {
	if (!current_pipeline || !current_pipeline->impl.program) return;
	gpu_constant_buffer_unlock(&constant_buffer);
	gpu_set_constant_buffer(&constant_buffer, constant_buffer_index * GPU_CONSTANT_BUFFER_SIZE, GPU_CONSTANT_BUFFER_SIZE);
	gpu_draw_internal();

	constant_buffer_index++;
	if (constant_buffer_index >= GPU_CONSTANT_BUFFER_MULTIPLE) constant_buffer_index = 0;
	gpu_constant_buffer_lock(&constant_buffer, constant_buffer_index * GPU_CONSTANT_BUFFER_SIZE, GPU_CONSTANT_BUFFER_SIZE);
}

void gpu_barrier(gpu_texture_t *render_target, gpu_texture_state_t state) {}

void gpu_get_render_target_pixels(gpu_texture_t *render_target, uint8_t *data) {
	if (!render_target || !render_target->impl.fbo) return;
	glBindFramebuffer(GL_FRAMEBUFFER, render_target->impl.fbo);
	GLint internal; GLenum fmt, type;
	gl_format(render_target->format, &internal, &fmt, &type);
	glReadPixels(0, 0, render_target->width, render_target->height, fmt, type, data);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GLCHECK();
}

void gpu_create_framebuffers(int depth_buffer_bits) {
	framebuffer_depth.format = GPU_TEXTURE_FORMAT_D32;
	framebuffer_depth.width = window_width;
	framebuffer_depth.height = window_height;
	gpu_render_target_init2(&framebuffer_depth, window_width, window_height, GPU_TEXTURE_FORMAT_D32, 0);

	for (int i = 0; i < GPU_FRAMEBUFFER_COUNT; ++i) {
		framebuffers[i].width = window_width;
		framebuffers[i].height = window_height;
		framebuffers[i].format = GPU_TEXTURE_FORMAT_RGBA32;
		framebuffers[i].impl.is_render_target = true;
		framebuffers[i].impl.width = window_width;
		framebuffers[i].impl.height = window_height;
	}
}

bool gpu_init_internal(int depth_buffer_bits, bool vsync) {
	if (!egl_init()) return false;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glFrontFace(GL_CCW);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	gpu_create_framebuffers(depth_buffer_bits);
	iron_log("GL: context ready %dx%d", window_width, window_height);
	return true;
}

bool gpu_init(int depth_buffer_bits, bool vsync) {
	return gpu_init_internal(depth_buffer_bits, vsync);
}

void gpu_set_int(int location, int value) { glUniform1i(location, value); GLCHECK(); }
void gpu_set_int2(int location, int v1, int v2) { glUniform2i(location, v1, v2); GLCHECK(); }
void gpu_set_int3(int location, int v1, int v2, int v3) { glUniform3i(location, v1, v2, v3); GLCHECK(); }
void gpu_set_int4(int location, int v1, int v2, int v3, int v4) { glUniform4i(location, v1, v2, v3, v4); GLCHECK(); }
void gpu_set_ints(int location, int *values, int count) { GLCHECK(); }
void gpu_set_float(int location, float value) { glUniform1f(location, value); GLCHECK(); }
void gpu_set_float2(int location, float v1, float v2) { glUniform2f(location, v1, v2); GLCHECK(); }
void gpu_set_float3(int location, float v1, float v2, float v3) { glUniform3f(location, v1, v2, v3); GLCHECK(); }
void gpu_set_float4(int location, float v1, float v2, float v3, float v4) { glUniform4f(location, v1, v2, v3, v4); GLCHECK(); }
void gpu_set_floats(int location, f32_array_t *values) { GLCHECK(); }
void gpu_set_bool(int location, bool value) { glUniform1i(location, value); GLCHECK(); }
void gpu_set_mat3(int location, mat3_t value) { GLCHECK(); }
void gpu_set_mat4(int location, mat4_t value) { GLCHECK(); }

uint32_t gpu_texture_format_size(gpu_texture_format_t format) {
	switch (format) {
	case GPU_TEXTURE_FORMAT_RGBA32: return 4;
	case GPU_TEXTURE_FORMAT_RGBA64: return 8;
	case GPU_TEXTURE_FORMAT_RGBA128: return 16;
	case GPU_TEXTURE_FORMAT_R8: return 1;
	case GPU_TEXTURE_FORMAT_R16: return 2;
	case GPU_TEXTURE_FORMAT_R32: return 4;
	case GPU_TEXTURE_FORMAT_D32: return 4;
	case GPU_TEXTURE_FORMAT_RGBA32_BC7: return 1;
	default: return 4;
	}
}

