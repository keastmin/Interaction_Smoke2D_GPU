#ifndef __DENSITY_H__
#define __DENSITY_H__

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class density {
	glm::vec3* d_dens_buffer;
	glm::vec3* d_dens_color_buffer;

	GLuint densitybuffer;
	cudaGraphicsResource* cudaVBODens;
	size_t numByteDens;

	GLuint densitycolorbuffer;
	cudaGraphicsResource* cudaVBODensColor;
	size_t numByteDensColor;
public:
	density(int N, double dx, double dy);
	~density();

public:
	void init(int N, double dx, double dy);
	void draw_dens(int N, double* kd, int* collision_result);
};

__global__ void init_dens(int N, glm::vec3* dens, double dx, double dy);
__global__ void init_color_dens(int N, glm::vec3* densC);
__global__ void update_dens(int N, glm::vec3* densC, double* kd, int* collision_result);

#endif __DENSITY_H__