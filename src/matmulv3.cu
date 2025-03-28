#include "matmul.cuh"

__global__ void matmulv3(float *A, float *B, float *C, int N) {
    int nx = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    int ny = (blockIdx.y * blockDim.y + threadIdx.y) * V;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    
    __shared__ float s_a[S][L];
    __shared__ float s_b[S][L];

    float c[V][V] = {0};

    for (int ko = 0; ko < N; ko += S) {
        int round = S * L / (blockDim.x * blockDim.y);
        int stride = blockDim.x * blockDim.y / S;
        for (int i = 0; i < round; i++) {
            int sx = tid % S;
            int sy = tid / S + i * stride;
            int ax = ko + sx;
            int ay = blockIdx.y * L + sy;
            int bx = blockIdx.x * L + sy;
            int by = ko + sx;
            s_a[sx][sy] = A[ay * N + ax];
            s_b[sx][sy] = B[by * N + bx];
        }
        __syncthreads();

        for (int ki = 0; ki < S; ki++) {
            float a[V];
            float b[V];
            for (int i = 0; i < V; i++) {
                a[i] = s_a[ki][threadIdx.y * V + i];
                b[i] = s_b[ki][threadIdx.x * V + i];
            }
            for (int y = 0; y < V; y++) {
                for (int x = 0; x < V; x++) {
                    c[y][x] += a[y] * b[x];
                }
            }
        }
        __syncthreads();
    }
    for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
            C[(ny + y) * N + (nx + x)] = c[y][x];
        }
    }
}