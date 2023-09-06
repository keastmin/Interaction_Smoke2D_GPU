#ifndef __DRAWSPHERE_H__
#define __DRAWSPHERE_H__

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class drawSphere {
	int numStacks;
	int numSlices;
	int sphereNum;

	int* d_collision_result;

	GLuint spherebuffer;
	cudaGraphicsResource* cudaVBOsphere;
	size_t numBytesphere;

	GLuint spereColorBuffer;
	glm::vec3* sphereColors;

	glm::vec3* d_sphere_buffer;
	glm::vec3* d_color_buffer;

	double xpos, ypos;
	double sphereScale;
public:
	drawSphere(int N);
	~drawSphere();

public:
	void init_camera();
	void init(int N);
	void drawSph(int N);
};

__global__ void init_sphere(int stacks, int slices, glm::vec3* sphere, double x, double y, double scale);
__global__ void check_collision();

#endif __DRAWSPHERE_H__