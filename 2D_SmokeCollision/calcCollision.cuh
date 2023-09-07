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
	int* draw_InsideCell;
	int* draw_OutsideCell;
	int* calc_InsideCell;
	int* calc_OutsideCell;
	int* collisionResult_D;
	int* collisionResult_IX;

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
__global__ void divide_collision_draw(int N, int* insideCell, int* outsideCell, int* drawResult);

#endif __CALCCOLLISION_H__