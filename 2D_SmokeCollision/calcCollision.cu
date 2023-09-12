#include "calcCollision.cuh"

#define IX(i, j) ((i) + (N+2)*(j))
#define DIX(i, j) ((i) + (N)*(j))

calcCollision::calcCollision(int N, double dx, double dy, float scale) {
	init(N, dx, dy, scale);
}

calcCollision::~calcCollision() {
	cudaFree(collisionResult_D);
	cudaFree(collisionResult_IX);
}

void calcCollision::init(int N, double dx, double dy, float scale) {
	cx = dx;
	cy = dy;
	cScale = scale;


	// 충돌 감지 셀 초기화
	cudaMalloc((void**)&collisionResult_D, N  * N * sizeof(int));
	cudaMalloc((void**)&collisionResult_IX, (N + 2) * (N + 2) * sizeof(int));

	cudaMemset(collisionResult_D, 0, N * N * sizeof(int));
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

// 외부 셀 정의
__global__ void divide_collision_draw(int N, int* drawResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = DIX(i, j);
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {  // 배열 경계를 벗어나지 않도록 수정
		if (drawResult[idx] == 1) {
			if (drawResult[DIX(i - 1, j)] == 0 || drawResult[DIX(i + 1, j)] == 0 ||
				drawResult[DIX(i, j - 1)] == 0 || drawResult[DIX(i, j + 1)] == 0) {
				drawResult[idx] = 2;  // 외부 셀
			}
		}
	}
	else if ((i == 0 && j <= N - 1) || (i == N - 1 && j <= N - 1) ||
			(i <= N - 1 && j == 0) || (i <= N - 1 && j == N - 1)) {		
		drawResult[idx] = (drawResult[idx] == 1) ? 2 : 0;
	}
}

__global__ void divide_collision_calc(int N, int* calcResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = IX(i + 1, j + 1);

	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {  // 배열 경계를 벗어나지 않도록 수정
		if (calcResult[idx] == 1) {
			if (calcResult[IX(i - 1, j)] == 0 || calcResult[IX(i + 1, j)] == 0 ||
				calcResult[IX(i, j - 1)] == 0 || calcResult[IX(i, j + 1)] == 0) {
				calcResult[idx] = 2;  // 외부 셀
			}
		}
	}
	else if ((i == 0 && j <= N - 1) || (i == N - 1 && j <= N - 1) ||
		(i <= N - 1 && j == 0) || (i <= N - 1 && j == N - 1)) {
		calcResult[idx] = (calcResult[idx] == 1) ? 2 : 0;
	}
}

// 경계조건 셀 정의
__global__ void divide_midCell_draw(int N, int* drawResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = DIX(i, j);
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		if (drawResult[idx] == 1 &&
			(drawResult[DIX(i - 1, j)] == 2 || drawResult[DIX(i + 1, j)] == 2 ||
			drawResult[DIX(i, j - 1)] == 2 || drawResult[DIX(i, j + 1)] == 2 ||
			drawResult[DIX(i + 1, j + 1)] == 2 || drawResult[DIX(i + 1, j - 1)] == 2 || 
			drawResult[DIX(i - 1, j + 1)] == 2 || drawResult[DIX(i - 1, j - 1)] == 2)) {
			drawResult[idx] = 3;
		}
	}
}

// 경계조건 셀 설정 - 3 : 위, 4 : 아래, 5 : 오른쪽, 6 : 왼쪽, 7 : 왼쪽 위 모서리, 8 : 오른쪽 위 모서리, 9 : 오른쪽 아래 모서리, 10 : 왼쪽 아래 모서리
// 면에 대한 셀 - 3 : 위, 4 : 아래, 5 : 오른쪽, 6 : 왼쪽
__global__ void divide_midCell_calc(int N, int* calcResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = IX(i + 1, j + 1);
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && calcResult[idx] == 1) {
		if (calcResult[IX(i, j + 1)] == 2) {
			calcResult[idx] = 3;
		}
		else if (calcResult[IX(i, j - 1)] == 2) {
			calcResult[idx] = 4;
		}
		else if (calcResult[IX(i + 1, j)] == 2) {
			calcResult[idx] = 5;
		}
		else if (calcResult[IX(i - 1, j)] == 2) {
			calcResult[idx] = 6;
		}
	}
}

// 외부 모서리에 대한 셀 - 7 : 왼쪽 위 모서리, 8 : 오른쪽 위 모서리, 9 : 오른쪽 아래 모서리, 10 : 왼쪽 아래 모서리
__global__ void divide_OutCornerCell_calc(int N, int* calcResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = IX(i + 1, j + 1);
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && calcResult[idx] > 2) {
		if (calcResult[IX(i - 1, j)] == 2 && calcResult[IX(i, j + 1)] == 2) {
			calcResult[idx] = 7;
		}
		else if (calcResult[IX(i + 1, j)] == 2 && calcResult[IX(i, j + 1)] == 2) {
			calcResult[idx] = 8;
		}
		else if (calcResult[IX(i + 1, j)] == 2 && calcResult[IX(i, j - 1)] == 2) {
			calcResult[idx] = 9;
		}
		else if (calcResult[IX(i - 1, j)] == 2 && calcResult[IX(i, j - 1)] == 2) {
			calcResult[idx] = 10;
		}
	}
}

// 내부 모서리에 대한 셀 - 11 : 왼쪽 위 모서리, 12 오른쪽 위 모서리, 13 오른쪽 아래 모서리, 14 : 왼쪽 아래 모서리
__global__ void divide_InCornerCell_calc(int N, int* calcResult) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = IX(i + 1, j + 1);
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && calcResult[idx] > 2) {
		if (calcResult[IX(i - 1, j)] > 2 && calcResult[IX(i, j + 1)] > 2) {
			calcResult[idx] = 11;
		}
		else if (calcResult[IX(i + 1, j)] > 2 && calcResult[IX(i, j + 1)] > 2) {
			calcResult[idx] = 12;
		}
		else if (calcResult[IX(i + 1, j)] > 2 && calcResult[IX(i, j - 1)] > 2) {
			calcResult[idx] = 13;
		}
		else if (calcResult[IX(i - 1, j)] > 2 && calcResult[IX(i, j - 1)] > 2) {
			calcResult[idx] = 14;
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

	divide_collision_draw<<<gridDim, blockDim>>>(N, collisionResult_D);
	divide_collision_calc<<<gridDim, blockDim>>>(N, collisionResult_IX);
	cudaDeviceSynchronize();

	divide_midCell_draw<<<gridDim, blockDim>>>(N, collisionResult_D);
	divide_midCell_calc<<<gridDim, blockDim>>>(N, collisionResult_IX);
	cudaDeviceSynchronize();

	divide_OutCornerCell_calc<<<gridDim, blockDim>>>(N, collisionResult_IX);
	divide_InCornerCell_calc<<<gridDim, blockDim>>>(N, collisionResult_IX);
}