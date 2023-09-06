#ifndef __VELOCITY_H__
#define __VELOCITY_H__
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class velocity {
	glm::vec3* d_static_vel_buffer;
	glm::vec3* d_dynamic_vel_buffer;
	glm::vec3* d_vel_buffer;
	glm::vec3* d_vel_color_buffer;

	GLuint velocitybuffer;
	GLuint velocitycolorbuffer;
	cudaGraphicsResource* cudaVBOVel;
	size_t numByteVel;

public:
	velocity(int N, double dx, double dy);
	~velocity();

public:
	void init(int N, double dx, double dy);
	void draw_velocity(int N, double* ku, double* kv);
};

__global__ void init_vel(int N, glm::vec3* vel, glm::vec3* stvel, glm::vec3* dyvel, double dx, double dy);
__global__ void update_vel(int N, glm::vec3* vel, glm::vec3* dyvel, double* ku, double* kv);

#endif __VELOCITY_H__