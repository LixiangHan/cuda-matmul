#include "matmul.cuh"

__global__ void matmulv1(float *A, float *B, float *C, int N) {
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;

    if (nx < N && ny < N) {
        float accumulator = 0.0f;
        float *a = A + ny * N;
        float *b = B + nx;
        for (int k = 0; k < N; k++) {
            accumulator += a[k] * b[k * N];
        }
        C[ny * N + nx] = accumulator;
    }
}