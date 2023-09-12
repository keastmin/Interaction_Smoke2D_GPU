#ifndef __CALCCOLLISION_H__
#define __CALCCOLLISION_H__

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "control.h"

class calcCollision {
public:
	int* collisionResult_D;
	int* collisionResult_IX;

	int* calc_CornerCell;

	double cx, cy;
	float cScale;
public:
	calcCollision(int N, double dx, double dy, float scale);
	~calcCollision();
public:
	void init(int N, double dx, double dy, float scale);
	void check_collision(int N);
};

__global__ void collision_kernel(int N, glm::vec3 sphere_center, float sphere_radius, int* drawResult, int* calcResult, double dx, double dy);
__global__ void divide_collision_draw(int N, int* drawResult);
__global__ void divide_collision_calc(int N, int* calcResult);
__global__ void divide_midCell_draw(int N, int* drawResult);
__global__ void divide_midCell_calc(int N, int* calcResult);
__global__ void divide_OutCornerCell_calc(int N, int* calcResult);
__global__ void divide_InCornerCell_calc(int N, int* calcResult);

#endif __CALCCOLLISION_H__