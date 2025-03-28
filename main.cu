#include <stdio.h>
#include <stdlib.h>
#include "common.cuh"
#include "matmul.cuh"
#include "utils.hpp"

const unsigned int N = 1024;

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
    matmulv0(A, B, ref_C, N);
    
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
    {
        CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEvent_t start, stop;
        float elapsed;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        dim3 block_size(BlockSize, BlockSize);
        dim3 grid_size((N + BlockSize - 1) / BlockSize, (N + BlockSize - 1) / BlockSize);

        CHECK(cudaEventRecord(start));

        matmulv1<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));

        CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        float error = check_result(C, ref_C, N);
        printf("---------- matmulv1 ----------\n");
        printf("  - time: %f ms\n", elapsed);
        printf("  - error: %f\n\n", error);
    }

    // Test matmulv2
    {
        CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEvent_t start, stop;
        float elapsed;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        dim3 block_size(BlockSize, BlockSize);
        dim3 grid_size((N + BlockSize - 1) / BlockSize / V, (N + BlockSize - 1) / BlockSize / V);

        CHECK(cudaEventRecord(start));
        
        matmulv2<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));

        CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        float error = check_result(C, ref_C, N);
        printf("---------- matmulv2 ----------\n");
        printf("  - time: %f ms\n", elapsed);
        printf("  - error: %f\n\n", error);
    }
    
    // Test matmulv3
    {
        CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEvent_t start, stop;
        float elapsed;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        dim3 block_size(BlockSize, BlockSize);
        dim3 grid_size((N + BlockSize - 1) / BlockSize / V, (N + BlockSize - 1) / BlockSize / V);

        CHECK(cudaEventRecord(start));

        matmulv3<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));

        CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        float error = check_result(C, ref_C, N);
        printf("---------- matmulv3 ----------\n");
        printf("  - time: %f ms\n", elapsed);
        printf("  - error: %f\n\n", error);
    }

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