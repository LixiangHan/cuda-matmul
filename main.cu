#include <stdio.h>
#include <stdlib.h>
#include "common.cuh"
#include "matmul.cuh"
#include "utils.hpp"

const unsigned int N = 4096;
const unsigned int BlockSize = 16;

int main(int argc, char **argv) {
    // Allocate host memory
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    float *ref_C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices
    random_matrix(A, N);
    random_matrix(B, N);

    // Compute reference result
    // matmulv0(A, B, ref_C, N);
    
    // Allocate device memory
    float *d_A;
    float *d_B;
    float *d_C;

    CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    CHECK(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Test matmulv1
    dim3 block_size(BlockSize, BlockSize);
    dim3 grid_size((N + BlockSize - 1) / BlockSize, (N + BlockSize - 1) / BlockSize);

    matmulv1<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    // float error = check_result(C, ref_C, N);
    // printf("matmulv1 error: %f\n", error);

    // Test matmulv2
    grid_size.x = (N + BlockSize - 1) / BlockSize / V;
    grid_size.y = (N + BlockSize - 1) / BlockSize / V;

    matmulv2<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    // float error = check_result(C, ref_C, N);
    // printf("matmulv2 error: %f\n", error);
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    free(ref_C);

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return 0;
}