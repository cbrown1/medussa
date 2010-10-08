#include <stdio.h>

typedef struct f_matrix {
    int m;
    int n;
    float *arr;
} f_matrix;

float *mat (f_matrix *, int, int);

int matrix_mult (float *a,  int a_m,  int a_n,
                 float *b,  int b_m,  int b_n,
                 float *ab, int ab_m, int ab_n);

void print_matrix (float *a, int a_m, int a_n);