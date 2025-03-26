#include "matmul.cuh"

void matmulv0(float *A, float *B, float *C, int N) {
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float *c = &C[y * N + x];
            float *a = &A[y * N];
            float *b = &B[x];
            for (int k = 0; k < N; k++) {
                *c += a[k] * b[k * N];
            }
        }
    }
}