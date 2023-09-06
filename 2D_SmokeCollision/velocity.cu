#include "velocity.cuh"
#define IX(i, j) ((i) + (N + 2) * (j))
#define DIX(i, j) ((i) + (N) * (j))

velocity::velocity(int N, double dx, double dy) {
	init(N, dx, dy);
}

velocity::~velocity() {
	cudaGraphicsUnregisterResource(cudaVBOVel);
	glDeleteBuffers(1, &velocitybuffer);
	glDeleteBuffers(1, &velocitycolorbuffer);
	cudaFree(d_static_vel_buffer);
	cudaFree(d_dynamic_vel_buffer);
}

__global__ void init_vel(int N, glm::vec3* vel, glm::vec3* stvel, glm::vec3* dyvel, double dx, double dy) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		int idx = DIX(i, j);
		double x, y, h;
		h = 1.0f / N;
		x = (i - 0.5) * h + dx;
		y = (j - 0.5) * h + dy;

		glm::vec3 initPos(x, y, 0.0);

		stvel[idx] = initPos;
		dyvel[idx] = initPos;

		vel[2 * idx + 0] = stvel[idx];
		vel[2 * idx + 1] = dyvel[idx];
	}
}

void velocity::init(int N, double dx, double dy) {
	int size = N * N;
	size_t d_size = size * sizeof(glm::vec3);
	cudaMalloc((void**)&d_static_vel_buffer, d_size);
	cudaMalloc((void**)&d_dynamic_vel_buffer, d_size);

	glGenBuffers(1, &velocitybuffer);
	glBindBuffer(GL_ARRAY_BUFFER, velocitybuffer);
	glBufferData(GL_ARRAY_BUFFER, 2 * d_size, NULL, GL_STREAM_DRAW);

	dim3 blockSize(16, 16);
	dim3 numBlock((N + blockSize.x - 1) / blockSize.x, (N  + blockSize.y - 1) / blockSize.y);
	cudaGraphicsGLRegisterBuffer(&cudaVBOVel, velocitybuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBOVel, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vel_buffer, &numByteVel, cudaVBOVel);
	init_vel<<<numBlock, blockSize>>>(N, d_vel_buffer, d_static_vel_buffer, d_dynamic_vel_buffer, dx, dy);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);

	d_vel_color_buffer = new glm::vec3[2 * size];
	glm::vec3 init_color(1.0, 1.0, 1.0);
	for (int i = 0; i < 2 * size; i++) {
		d_vel_color_buffer[i] = init_color;
	}

	glGenBuffers(1, &velocitycolorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, velocitycolorbuffer);
	glBufferData(GL_ARRAY_BUFFER, 2 * size * sizeof(glm::vec3), d_vel_color_buffer, GL_STATIC_DRAW);
}

__global__ void update_vel(int N, glm::vec3 * vel, glm::vec3 * dyvel, double* ku, double* kv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		int idx = DIX(i, j);
		int velIdx = IX(i + 1, j + 1);

		vel[2 * idx + 1].x = dyvel[idx].x + ku[velIdx];
		vel[2 * idx + 1].y = dyvel[idx].y + kv[velIdx];
	}
}

void velocity::draw_velocity(int N, double* ku, double* kv) {
	dim3 blockSize(16, 16);
	dim3 numBlock((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	cudaGraphicsMapResources(1, &cudaVBOVel, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vel_buffer, &numByteVel, cudaVBOVel);
	update_vel<<<numBlock, blockSize>>>(N, d_vel_buffer, d_static_vel_buffer, ku, kv);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, velocitybuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, velocitycolorbuffer);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glDrawArrays(GL_LINES, 0, 2 * N * N);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}