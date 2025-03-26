#include <stdio.h>

#define CHECK(call) \
do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error: %s:%d, ", __FILE__, __LINE__); \
        printf("    code: %d\n", error); \
        printf("    info: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)
