#include "calcCollision.cuh"

#define IX(i, j) ((i) + (N+2)*(j))
#define DIX(i, j) ((i) + (N)*(j))

calcCollision::calcCollision(int N, double dx, double dy, float scale) {
	init(N, dx, dy, scale);
}

calcCollision::~calcCollision() {
	cudaFree(draw_InsideCell);
	cudaFree(draw_OutsideCell);
	cudaFree(collisionResult_D);
	cudaFree(calc_InsideCell);
	cudaFree(calc_OutsideCell);
	cudaFree(collisionResult_IX);
}

void calcCollision::init(int N, double dx, double dy, float scale) {
	cx = dx;
	cy = dy;
	cScale = scale;

	cudaMalloc((void**)&draw_InsideCell, N * N * sizeof(int));
	cudaMalloc((void**)&draw_OutsideCell, N * N * sizeof(int));
	cudaMalloc((void**)&collisionResult_D, N  * N * sizeof(int));
	cudaMalloc((void**)&calc_InsideCell, (N + 2) * (N + 2) * sizeof(int));
	cudaMalloc((void**)&calc_OutsideCell, (N + 2) * (N + 2) * sizeof(int));
	cudaMalloc((void**)&collisionResult_IX, (N + 2) * (N + 2) * sizeof(int));

	cudaMemset(draw_InsideCell, 0, N * N * sizeof(int));
	cudaMemset(draw_OutsideCell, 0, N * N * sizeof(int));
	cudaMemset(collisionResult_D, 0, N * N * sizeof(int));
	cudaMemset(calc_InsideCell, 0, (N + 2) * (N + 2) * sizeof(int));
	cudaMemset(calc_OutsideCell, 0, (N + 2) * (N + 2) * sizeof(int));
	cudaMemset(collisionResult_IX, 0, (N + 2) * (N + 2) * sizeof(int));
}

__global__ void collision_kernel(int N, glm::vec3 sphere_center, float sphere_radius, int* drawResult, int* calcResult, double dx, double dy) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		int didx = DIX(i, j);
		int idx = IX(i + 1, j + 1);
		double h, x, y;
		h = 1.0 / N;
		x = (i - 0.5) * h + dx;
		y = (j - 0.5) * h + dy;

		glm::vec3 cell_center(x, y, 0.0f);

		// 구체와 셀 중심점 간의 거리 계산
		float distance = glm::length(cell_center - sphere_center);

		// 충돌 감지
		if (distance <= sphere_radius) {
			drawResult[didx] = 1;  // 충돌 발생
			calcResult[idx] = 1;
		}
		else {
			drawResult[didx] = 0;  // 충돌 없음
			calcResult[idx] = 0;
		}
	}
}

__global__ void divide_collision_draw(int N, int* insideCell, int* outsideCell, int* drawResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {  // 배열 경계를 벗어나지 않도록 수정
		int idx = DIX(i, j);
		if (drawResult[idx] == 1) {
			if (drawResult[DIX(i - 1, j)] == 0 || drawResult[DIX(i + 1, j)] == 0 ||
				drawResult[DIX(i, j - 1)] == 0 || drawResult[DIX(i, j + 1)] == 0) {
				drawResult[idx] = 2;  // 외부 셀
			}
			else {
				drawResult[idx] = 1;  // 내부 셀
			}
		}
		else {
			drawResult[idx] = 0;  // 충돌이 일어나지 않은 셀
		}
	}
}

void calcCollision::check_collision(int N) {
	glm::vec3 cameraPos = getCameraPosition();
	glm::vec3 cameraFront = getCameraDirection();
	float t = -cameraPos.z / cameraFront.z;
	glm::vec3 pointInWorld = cameraPos + t * cameraFront;
	double xpos = pointInWorld.x;
	double ypos = pointInWorld.y;

	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

	glm::vec3 sphere_center(xpos, ypos, 0.0f);
	collision_kernel<<<gridDim, blockDim>>>(N, sphere_center, cScale, collisionResult_D, collisionResult_IX, cx, cy);
	cudaDeviceSynchronize();

	divide_collision_draw<<<gridDim, blockDim>>>(N, draw_InsideCell, draw_OutsideCell, collisionResult_D);
}