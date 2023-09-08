#ifndef __FLUIDSOLVER_CUH__
#define __FLUIDSOLVER_CUH__

#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "calcCollision.cuh"

__global__ void add_source(int N, double* x, double* s, double dt);
void set_bnd(int N, int b, double* x, int* calcIdx);
void lin_solve(int N, int b, double* x, double* x0, double a, double c, int* calcIdx);
void diffuse(int N, int b, double* x, double* x0, double diff, double dt, int* calcIdx);
void advect(int N, int b, double* d, double* d0, double* u, double* v, double dt, int* calcIdx);
void project(int N, double* u, double* v, double* p, double* div, int* calcIdx);
void dens_step(int N, double* x, double* x0, double* u, double* v, double diff, double dt, int* calcIdx);
void vel_step(int N, double* u, double* v, double* u0, double* v0, double visc, double dt, int* calcIdx);


#endif __FLUIDSOLVER_CUH__