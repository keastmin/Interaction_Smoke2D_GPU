#include "drawSphere.cuh"
#include "control.h"

#define M_PI 3.141592

drawSphere::drawSphere(int N) {
	init(N);
}

drawSphere::~drawSphere() {
	delete sphereColors;
	glDeleteBuffers(1, &spherebuffer);
	glDeleteBuffers(1, &spereColorBuffer);
	cudaFree(d_collision_result);
	cudaGraphicsUnregisterResource(cudaVBOsphere);
}

__global__ void init_sphere(int stacks, int slices, glm::vec3* sphere, double x, double y, double scale) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < stacks && j < slices) {
		float stackInterval = M_PI / (float)stacks;
		float sliceInterval = 2.0 * M_PI / (float)slices;

		float stackAngle1 = i * stackInterval;
		float stackAngle2 = (i + 1) * stackInterval;

		float sliceAngle1 = j * sliceInterval;
		float sliceAngle2 = (j + 1) * sliceInterval;
			
		glm::vec3 vertex1 = glm::vec3(
			x + scale * sinf(stackAngle1) * cosf(sliceAngle1),
			y + scale * cosf(stackAngle1),
			scale * sinf(stackAngle1) * sinf(sliceAngle1)
		);

		glm::vec3 vertex2 = glm::vec3(
			x + scale * sinf(stackAngle2) * cosf(sliceAngle1),
			y + scale * cosf(stackAngle2),
			scale * sinf(stackAngle2) * sinf(sliceAngle1)
		);

		glm::vec3 vertex3 = glm::vec3(
			x + scale * sinf(stackAngle1) * cosf(sliceAngle2),
			y + scale * cosf(stackAngle1),
			scale * sinf(stackAngle1) * sinf(sliceAngle2)
		);

		glm::vec3 vertex4 = glm::vec3(
			x + scale * sinf(stackAngle2) * cosf(sliceAngle2),
			y + scale * cosf(stackAngle2),
			scale * sinf(stackAngle2) * sinf(sliceAngle2)
		);

		int index = (i * slices + j) * 6;
		sphere[index + 0] = vertex1;
		sphere[index + 1] = vertex2;
		sphere[index + 2] = vertex3;

		sphere[index + 3] = vertex2;
		sphere[index + 4] = vertex4;
		sphere[index + 5] = vertex3;
	}
}

void drawSphere::init_camera() {
	// 구체 시작 위치
	glm::vec3 cameraPos = getCameraPosition();
	glm::vec3 cameraFront = getCameraDirection();
	float t = -cameraPos.z / cameraFront.z;
	glm::vec3 pointInWorld = cameraPos + t * cameraFront;
	xpos = pointInWorld.x;
	ypos = pointInWorld.y;
}

void drawSphere::init(int N) {
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

	numStacks = 20;
	numSlices = 20;
	sphereNum = 6 * numStacks * numSlices;
	sphereScale = 0.05;

	init_camera();

	// 구체 버퍼
	glGenBuffers(1, &spherebuffer);
	glBindBuffer(GL_ARRAY_BUFFER, spherebuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec3), NULL, GL_STATIC_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBOsphere, spherebuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBOsphere, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_sphere_buffer, &numBytesphere, cudaVBOsphere);
	init_sphere<<<gridDim, blockDim>>>(numStacks, numSlices, d_sphere_buffer, xpos, ypos, sphereScale);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOsphere, 0);

	sphereColors = new glm::vec3[sphereNum];
	for (int i = 0; i < sphereNum; ++i) {
		sphereColors[i] = glm::vec3(0.0f, 1.0f, 0.0f);
	}

	glGenBuffers(1, &spereColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, spereColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec3), sphereColors, GL_STATIC_DRAW);

	// 충돌 감지 버퍼
	cudaMalloc((void**)&d_collision_result, N * N * sizeof(int));
	cudaMemset(d_collision_result, 0, N * N * sizeof(int));
}

void drawSphere::drawSph(int N) {
	glm::vec3 cameraPos = getCameraPosition();
	glm::vec3 cameraFront = getCameraDirection();
	float t = -cameraPos.z / cameraFront.z;
	glm::vec3 pointInWorld = cameraPos + t * cameraFront;
	xpos = pointInWorld.x;
	ypos = pointInWorld.y;

	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

	cudaGraphicsMapResources(1, &cudaVBOsphere, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_sphere_buffer, &numBytesphere, cudaVBOsphere);
	init_sphere << <gridDim, blockDim >> > (numStacks, numSlices, d_sphere_buffer, xpos, ypos, sphereScale);
	cudaGraphicsUnmapResources(1, &cudaVBOsphere, 0);

	glBindBuffer(GL_ARRAY_BUFFER, spherebuffer);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glBindBuffer(GL_ARRAY_BUFFER, spereColorBuffer);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);
	glDrawArrays(GL_TRIANGLES, 0, sphereNum);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}