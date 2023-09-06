#include <iostream>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "control.h"
#include "shader.h"
#include "FluidSolver.cuh"
#include "velocity.cuh"
#include "density.cuh"
#include "drawSphere.cuh"

velocity* _vel;
density* _den;
drawSphere* _sphere;

// 2차원 인덱스를 1차원 인덱스처럼 관리
#define IX(i, j) ((i) + (N+2)*(j))
#define DIX(i, j) ((i) + (N)*(j))

// 그리드 크기
#define SIZE 300

// 화면 크기
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

GLFWwindow* window;

// 솔버에 사용될 GPU 메모리 할당 변수
static double* u, * v, * u_prev, * v_prev;
static double* dens, * dens_prev;

// 솔버에 사용될 상수 데이터
static const int N = SIZE;
static double dt = 0.08;
static double diff = 0.0;
static double visc = 0.0;
static double force = 10.0;
static double source = 50.0f;

// 시뮬레이션 제어 변수
static int addforce[3] = { 0, 0, 0 };
static int mode = 0;

static int width = WINDOW_WIDTH;
static int height = WINDOW_HEIGHT;

// 데이터 소멸
void free_data() {
	if (u) cudaFree(u);
	if (v) cudaFree(v);
	if (u_prev) cudaFree(u_prev);
	if (v_prev) cudaFree(v_prev);
	if (dens) cudaFree(dens);
	if (dens_prev) cudaFree(dens_prev);
}

/* --------------------데이터 초기화-------------------- */
// 데이터 초기값 삽입 커널 함수
__global__ void initArray(double* array, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		array[i] = 0.0;
	}
}

// 초기화 커널 구동 함수
static void init_data() {
	int size = (N + 2) * (N + 2);
	size_t d_size = size * sizeof(double);

	cudaMalloc((void**)&u, d_size);
	cudaMalloc((void**)&v, d_size);
	cudaMalloc((void**)&u_prev, d_size);
	cudaMalloc((void**)&v_prev, d_size);
	cudaMalloc((void**)&dens, d_size);
	cudaMalloc((void**)&dens_prev, d_size);

	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initArray<<<numBlocks, blockSize>>>(u, size);
	initArray<<<numBlocks, blockSize>>>(v, size);
	initArray<<<numBlocks, blockSize>>>(u_prev, size);
	initArray<<<numBlocks, blockSize>>>(v_prev, size);
	initArray<<<numBlocks, blockSize>>>(dens, size);
	initArray<<<numBlocks, blockSize>>>(dens_prev, size);
}
/* ---------------------------------------------------- */

/* ------------------소스항 추가 함수------------------ */
__global__ void setForceAndSource(double* d, double* v, int i1, int j1, double forceValue, int i2, int j2, double sourceValue) {
	v[IX(i1, j1)] = forceValue;
	d[IX(i2, j2)] = sourceValue;
}

void get_force_source(double* d, double* u, double* v) {
	int i, j, size = (N + 2) * (N + 2);
	cudaMemset(u, 0, size * sizeof(double));
	cudaMemset(v, 0, size * sizeof(double));
	cudaMemset(d, 0, size * sizeof(double));

	double forceValue;
	double sourceValue;

	if (addforce[0] == 1) {
		i = N / 2;
		j = 2;

		if (i < 1 || i > N || j < 1 || j > N) {
			std::cerr << "범위 벗어남" << '\n';
			return;
		}

		forceValue = force * 3;
		sourceValue = source;
		setForceAndSource<<<1, 1>>>(d, v, i, j, forceValue, i, 10, sourceValue);
	}
}
/* --------------------------------------------------- */

// 시뮬레이션 구동 함수
void sim_fluid() {
	get_force_source(dens_prev, u_prev, v_prev);
	vel_step(N, u, v, u_prev, v_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, diff, dt);
	cudaDeviceSynchronize();
}

// 키보드 콜백 함수
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_Z && action == GLFW_RELEASE) {
		addforce[0] = (addforce[0] == 0) ? 1 : 0;
		std::cout << "addforce[0] : " << addforce[0] << '\n';
	}

	if (key == GLFW_KEY_X && action == GLFW_RELEASE) {
		addforce[1] = (addforce[1] == 0) ? 1 : 0;
		std::cout << "addforce[1] : " << addforce[1] << '\n';
	}

	if (key == GLFW_KEY_1 && action == GLFW_RELEASE) {
		mode = 0;
		std::cout << "mode : " << mode << '\n';
	}

	if (key == GLFW_KEY_2 && action == GLFW_RELEASE) {
		mode = 1;
		std::cout << "mode : " << mode << '\n';
	}
}

int main() {
	// GLFW 초기화
	if (!glfwInit()) {
		std::cerr << "GLFW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(width, height, "collision test", NULL, NULL);
	if (window == NULL) {
		std::cerr << "GLFW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "GLFW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}

	// 변수 초기화
	init_data();
	cudaDeviceSynchronize();

	// 속도, 밀도 클래스 초기화
	double drawX = -0.5f;
	double drawY = -0.5f;
	_vel = new velocity(N, drawX, drawY);
	_den = new density(N, drawX, drawY);
	_sphere = new drawSphere(N);

	// 쉐이더 읽기
	GLuint programID = LoadShaders("VertexShaderSL.txt", "FragmentShaderSL.txt");
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	GLuint alpValue = glGetUniformLocation(programID, "alphaValue");

	// 마우스 세팅
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwPollEvents();
	glfwSetCursorPos(window, width / 2, height / 2);

	cudaSetDevice(0);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	glfwSetKeyCallback(window, key_callback);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(programID);

		// 화면 이동, 컨트롤 control.h
		computeMatricesFromInputs(window, width, height);
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// 시뮬레이션 반복
		sim_fluid();

		glUniform1f(alpValue, 1.0f);
		if (mode == 0) {
			_den->draw_dens(N, dens);
		}
		if (mode == 1) {
			_vel->draw_velocity(N, u, v);
		}

		glUniform1f(alpValue, 0.3f);
		_sphere->drawSph(N);

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while ((glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0));

	// 데이터 정리
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glfwDestroyWindow(window);
	free_data();
	delete _vel;
	glfwTerminate();

	return 0;
}