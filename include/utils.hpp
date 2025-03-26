#pragma once
#include <stdio.h>
#include <math.h>
#include <random>

void random_matrix(float *A, int N);

void print_matrix(float *A, int N, const char *name);

float check_result(float *output, float *reference, int N);