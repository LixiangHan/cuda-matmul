#include "matmul.cuh"

__global__ void matmulv2(float *A, float *B, float *C, int N) {
    int nx = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    int ny = (blockIdx.y * blockDim.y + threadIdx.y) * V;

    if (nx + V < N  && ny + V < N) {
        float a[V];
        float b[V];
        float c[V][V] = {0};
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < V; i++) {
                a[i] = A[(ny + i) * N + k];
                b[i] = B[k * N + nx + i];
            }
            for (int yi = 0; yi < V; yi++) {
                for (int xi = 0; xi < V; xi++) {
                    c[yi][xi] += a[yi] * b[xi];
                }
            }
        }
        for (int yi = 0; yi < V; yi++) {
            for (int xi = 0; xi < V; xi++) {
                C[(ny + yi) * N + (nx + xi)] = c[yi][xi];
            }
        }
    }
}