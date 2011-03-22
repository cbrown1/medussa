#include <stdio.h>

typedef struct f_matrix {
    int m;
    int n;
    float *arr;
} f_matrix;

float *mat (f_matrix *, int, int);

int fmatrix_mult (float  *a,  int a_m,  int a_n,
                  float  *b,  int b_m,  int b_n,
                  float  *ab, int ab_m, int ab_n);

int dmatrix_mult (double *a,  int a_m,  int a_n,
                  double *b,  int b_m,  int b_n,
                  double *ab, int ab_m, int ab_n);

int dmatrix_mult_tof (double *a,  int a_m,  int a_n,
                      double *b,  int b_m,  int b_n,
                      float  *ab, int ab_m, int ab_n);

void fprint_matrix (float  *a, int a_m, int a_n);
void dprint_matrix (double *a, int a_m, int a_n);