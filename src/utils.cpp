#include "utils.hpp"

void random_matrix(float *A, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N * N; i++) {
        A[i] = dis(gen);
    }
}

void print_matrix(float *A, int N, const char *name) {
    printf("%s = [\n", name);
    for (int y = 0; y < N; y++) {
        printf("  ");
        for (int x = 0; x < N; x++) {
            printf("%5.2f", A[y * N + x]);
        }
        printf("\n");
    }
    printf("]\n");
}

float check_result(float *output, float *reference, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabs(output[i] - reference[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}
    