#include "FluidSolver.cuh"

#define IX(i,j) ((i)+(N+2)*(j))
#define LINEARSOLVERTIMES 10
#define SWAP(x0,x) {double * tmp=x0;x0=x;x=tmp;}

// 소스항 추가 커널
__global__ void add_source(int N, double* x, double* s, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (N + 2) * (N + 2);
    if (idx < size)
        x[idx] += dt * s[idx];
}

/* -------------------그리드 경계 조건------------------- */
// 그리드 면 경계 조건 커널
__global__ void k_set_bnd(int N, int b, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N) {
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
}

// 모서리 경계 조건 커널
__global__ void k_update_corners(int N, double* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
        x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
        x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    }
}

// 그리드 경계 조건 커널 구동 함수
void set_bnd(int N, int b, double* x)
{
    int blockSize = 256;
    int numBlock = (N + blockSize - 1) / blockSize;
    k_set_bnd << <numBlock, blockSize >> > (N, b, x);

    k_update_corners << <1, 1 >> > (N, x);
}
/* ----------------------------------------------------- */

/* -----------red black gauss seidel 선형 풀이----------- */
// red cell 커널
__global__ void red_cell_lin(int N, double* x, double* x0, double a, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N && (i + j) % 2 == 0) {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

// black cell 커널
__global__ void black_cell_lin(int N, double* x, double* x0, double a, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N && (i + j) % 2 == 1) {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

void lin_solve(int N, int b, double* x, double* x0, double a, double c) {
    int l;
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    for (l = 0; l < LINEARSOLVERTIMES; l++) {
        red_cell_lin << <gridDim, blockDim >> > (N, x, x0, a, c);
        cudaDeviceSynchronize();
        black_cell_lin << <gridDim, blockDim >> > (N, x, x0, a, c);
        cudaDeviceSynchronize();
        set_bnd(N, b, x);
        cudaDeviceSynchronize();
    }
}
/* ----------------------------------------------------- */

// 확산 함수
void diffuse(int N, int b, double* x, double* x0, double diff, double dt)
{
    double a = dt * diff * N * N;
    lin_solve(N, b, x, x0, a, 1 + 4 * a);
}

/* ----------------------이류 함수---------------------- */
// 이류 커널 함수
__global__ void k_advect(int N, double* d, double* d0, double* u, double* v, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= N && j <= N) {
        int i0, j0, i1, j1;
        double x, y, s0, t0, s1, t1, dtx, dty;

        dtx = dty = dt * N;
        x = i - dtx * u[IX(i, j)]; y = j - dty * v[IX(i, j)];
        if (x < 0.5f) x = 0.5; if (x > N + 0.5) x = N + 0.5; i0 = (int)x; i1 = i0 + 1;
        if (y < 0.5f) y = 0.5; if (y > N + 0.5) y = N + 0.5; j0 = (int)y; j1 = j0 + 1;

        s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;

        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}


// 이류 커널 구동 함수
void advect(int N, int b, double* d, double* d0, double* u, double* v, double dt)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    k_advect << <gridDim, blockDim >> > (N, d, d0, u, v, dt);
    cudaDeviceSynchronize();

    set_bnd(N, b, d);
    cudaDeviceSynchronize();
}
/* ---------------------------------------------------- */

/* ---------------------프로젝트 함수--------------------- */
// 발산을 계산하고 압력 필드를 0으로 초기화
__global__ void poison(int N, double* u, double* v, double* p, double* div) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= N && j <= N) {
        double h = 1.0 / N;
        div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
        p[IX(i, j)] = 0;
    }
}

// 속도 필드 업데이트 (질량 보존)
__global__ void k_project(int N, double* u, double* v, double* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= N && j <= N) {
        double h = 1.0 / N;
        u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
        v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    }
}

// 프로젝트 커널 함수를 구동하는 함수
void project(int N, double* u, double* v, double* p, double* div)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    poison<<<gridDim, blockDim>>>(N, u, v, p, div);
    cudaDeviceSynchronize();

    set_bnd(N, 0, div); set_bnd(N, 0, p);
    cudaDeviceSynchronize();

    lin_solve(N, 0, p, div, 1, 4);

    k_project<<<gridDim, blockDim>>>(N, u, v, p);
    cudaDeviceSynchronize();

    set_bnd(N, 1, u); set_bnd(N, 2, v);
    cudaDeviceSynchronize();
}
/* ------------------------------------------------------ */

// 밀도 필드 업데이트
void dens_step(int N, double* x, double* x0, double* u, double* v, double diff, double dt)
{
    // add source kernel
    int size = (N + 2) * (N + 2);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    add_source<<<numBlocks, blockSize>>>(N, x, x0, dt);
    cudaDeviceSynchronize();

    // swap and diffuse
    SWAP(x0, x); diffuse(N, 0, x, x0, diff, dt);
    SWAP(x0, x); advect(N, 0, x, x0, u, v, dt);
}

// 속도 필드 업데이트
void vel_step(int N, double* u, double* v, double* u0, double* v0, double visc, double dt)
{
    // add source kernel
    int size = (N + 2) * (N + 2);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    add_source<<<numBlocks, blockSize>>>(N, u, u0, dt); add_source<<<numBlocks, blockSize>>>(N, v, v0, dt);
    cudaDeviceSynchronize();

    // swap and diffuse
    SWAP(u0, u); diffuse(N, 1, u, u0, visc, dt);
    SWAP(v0, v); diffuse(N, 2, v, v0, visc, dt);

    // project and swap
    project(N, u, v, u0, v0);
    SWAP(u0, u); SWAP(v0, v);

    // advect
    advect(N, 1, u, u0, u0, v0, dt); advect(N, 2, v, v0, u0, v0, dt);

    // final project
    project(N, u, v, u0, v0);
}