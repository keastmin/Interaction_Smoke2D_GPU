#include "density.cuh"

#define IX(i, j) ((i) + (N+2) * (j))
#define DIX(i, j) ((i) + (N) * (j))

density::density(int N, double dx, double dy) {
	init(N, dx, dy);
}

density::~density() {
	glDeleteBuffers(1, &densitybuffer);
	glDeleteBuffers(1, &densitycolorbuffer);
	cudaGraphicsUnregisterResource(cudaVBODens);
	cudaGraphicsUnregisterResource(cudaVBODensColor);
}

__global__ void init_dens(int N, glm::vec3* dens, double dx, double dy) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		int idx = DIX(i, j);
		double x, y, h;
		h = 1.0 / N;
		x = (i - 1.0) * h + dx;
		y = (j - 1.0) * h + dy;

		glm::vec3 tmpd00(x, y, 0.0f);
		glm::vec3 tmpd01(x, y + h, 0.0f);
		glm::vec3 tmpd11(x + h, y + h, 0.0f);
		glm::vec3 tmpd10(x + h, y, 0.0f);

		dens[6 * idx + 0] = tmpd00;
		dens[6 * idx + 1] = tmpd01;
		dens[6 * idx + 2] = tmpd11;

		dens[6 * idx + 3] = tmpd11;
		dens[6 * idx + 4] = tmpd10;
		dens[6 * idx + 5] = tmpd00;
	}
}

__global__ void init_color_dens(int N, glm::vec3* densC) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		int idx = DIX(i, j);
		glm::vec3 initColor(0.0f, 0.0f, 0.0f);

		// 1번째 삼각형
		densC[6 * idx + 0] = initColor;
		densC[6 * idx + 1] = initColor;
		densC[6 * idx + 2] = initColor;
		// 2번째 삼각형
		densC[6 * idx + 3] = initColor;
		densC[6 * idx + 4] = initColor;
		densC[6 * idx + 5] = initColor;
	}
}

void density::init(int N, double dx, double dy) {
	dim3 blockSize(16, 16);
	dim3 numBlock((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	
	// 위치 버퍼
	glGenBuffers(1, &densitybuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densitybuffer);
	glBufferData(GL_ARRAY_BUFFER, 6 * N * N * sizeof(glm::vec3), NULL, GL_STATIC_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBODens, densitybuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODens, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_buffer, &numByteDens, cudaVBODens);
	init_dens << < numBlock, blockSize >> > (N, d_dens_buffer, dx, dy);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODens, 0);

	// 컬러 버퍼
	glGenBuffers(1, &densitycolorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densitycolorbuffer);
	glBufferData(GL_ARRAY_BUFFER, 6 * N * N * sizeof(glm::vec3), NULL, GL_STREAM_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBODensColor, densitycolorbuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_color_buffer, &numByteDensColor, cudaVBODensColor);
	init_color_dens << <numBlock, blockSize >> > (N, d_dens_color_buffer);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);
}


__global__ void update_dens(int N, glm::vec3* densC, double* kd, int* collision_result) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		double d00, d01, d11, d10;
		int idx = DIX(i, j);

		d00 = kd[IX(i + 1, j + 1)];
		d01 = kd[IX(i + 1, j + 2)];
		d11 = kd[IX(i + 2, j + 2)];
		d10 = kd[IX(i + 2, j + 1)];

		glm::vec3 densColord00(d00, d00, d00);
		glm::vec3 densColord01(d01, d01, d01);
		glm::vec3 densColord11(d11, d11, d11);
		glm::vec3 densColord10(d10, d10, d10);
		glm::vec3 collisionColorIn(1.0f, 0.0f, 0.0f);
		glm::vec3 collisionColorOut(1.0f, 1.0f, 0.0f);
		glm::vec3 collisionColorMid(1.0f, 0.0f, 1.0f);

		if (collision_result[idx] == 1) {
			densC[6 * idx + 0] = collisionColorIn;
			densC[6 * idx + 1] = collisionColorIn;
			densC[6 * idx + 2] = collisionColorIn;

			densC[6 * idx + 3] = collisionColorIn;
			densC[6 * idx + 4] = collisionColorIn;
			densC[6 * idx + 5] = collisionColorIn;
		}
		else if (collision_result[idx] == 2) {
			densC[6 * idx + 0] = collisionColorOut;
			densC[6 * idx + 1] = collisionColorOut;
			densC[6 * idx + 2] = collisionColorOut;

			densC[6 * idx + 3] = collisionColorOut;
			densC[6 * idx + 4] = collisionColorOut;
			densC[6 * idx + 5] = collisionColorOut;
		}
		else if (collision_result[idx] == 3) {
			densC[6 * idx + 0] = collisionColorMid;
			densC[6 * idx + 1] = collisionColorMid;
			densC[6 * idx + 2] = collisionColorMid;

			densC[6 * idx + 3] = collisionColorMid;
			densC[6 * idx + 4] = collisionColorMid;
			densC[6 * idx + 5] = collisionColorMid;
		}
		else {
			densC[6 * idx + 0] = densColord00;
			densC[6 * idx + 1] = densColord01;
			densC[6 * idx + 2] = densColord11;

			densC[6 * idx + 3] = densColord11;
			densC[6 * idx + 4] = densColord10;
			densC[6 * idx + 5] = densColord00;
		}
	}
}

void density::draw_dens(int N, double* kd, int* collision_result) {
	dim3 blockSize(16, 16);
	dim3 numBlock((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_color_buffer, &numByteDensColor, cudaVBODensColor);
	update_dens<<<numBlock, blockSize>>>(N, d_dens_color_buffer, kd, collision_result);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);

	glBindBuffer(GL_ARRAY_BUFFER, densitybuffer);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glBindBuffer(GL_ARRAY_BUFFER, densitycolorbuffer);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glDrawArrays(GL_TRIANGLES, 0, 6 * N * N);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

