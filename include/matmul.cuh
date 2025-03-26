#pragma once
#include <cuda_runtime.h>

const unsigned int V = 4;
const unsigned int S = 4;

void matmulv0(float *A, float *B, float *C, int N);

__global__ void matmulv1(float *A, float *B, float *C, int N);

__global__ void matmulv2(float *A, float *B, float *C, int N);

__global__ void matmulv3(float *A, float *B, float *C, int N);



