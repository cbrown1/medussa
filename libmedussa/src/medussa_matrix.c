#include "medussa_matrix.h"

float *mat (f_matrix *a, int i, int j)
{
    return &(a->arr[i * a->n + j]);
}

int fmatrix_mult (float *a,  int a_m,  int a_n,
                  float *b,  int b_m,  int b_n,
                  float *ab, int ab_m, int ab_n)
{
    int i, j, k;

    float ij;

    if (a_n != b_m) {
        // Matrix multiplication not defined
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        return -2;
    }

    for (i = 0; i < ab_m; i++) {
        for (j = 0; j < ab_n; j++) {
            ij = 0.0;

            for (k = 0; k < a_n; k++) {
                ij += a[i*a_n + k] * b[k*b_n + j];
            }

            ab[i*ab_n + j] = ij;
        }
    }
    return 0;
}


int dmatrix_mult (double *a,  int a_m,  int a_n,
                  double *b,  int b_m,  int b_n,
                  double *ab, int ab_m, int ab_n)
{
    int i, j, k;

    double ij;

    if (a_n != b_m) {
        // Matrix multiplication not defined
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        return -2;
    }

    for (i = 0; i < ab_m; i++) {
        for (j = 0; j < ab_n; j++) {
            ij = 0.0;

            for (k = 0; k < a_n; k++) {
                ij += a[i*a_n + k] * b[k*b_n + j];
            }
            // printf("i = %d, j = %d, ij = %.6f\n", i, j, ij);
            ab[i*ab_n + j] = ij;
        }
    }
    return 0;
}


int dmatrix_mult_tof (double *a,  int a_m,  int a_n,
                      double *b,  int b_m,  int b_n,
                      float  *ab, int ab_m, int ab_n)
{
    int i, j, k;

    double ij;

    if (a_n != b_m) {
        // Matrix multiplication not defined
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        return -2;
    }

    for (i = 0; i < ab_m; i++) {
        for (j = 0; j < ab_n; j++) {
            ij = 0.0;

            for (k = 0; k < a_n; k++) {
                ij += a[i*a_n + k] * b[k*b_n + j];
            }
            ab[i*ab_n + j] = (float) ij;
        }
    }

    return 0;
}


void fprint_matrix (float *a, int a_m, int a_n)
{
    int i, j;
    for (i = 0; i < a_m; i++) {
	    for (j = 0; j < a_n; j++) {
	        printf("%.6f ", a[i*a_n + j]);
	    }
	    printf("\n");
    }
}


void dprint_matrix (double *a, int a_m, int a_n)
{
    int i, j;
    for (i = 0; i < a_m; i++) {
	    for (j = 0; j < a_n; j++) {
	        printf("%.6f ", a[i*a_n + j]);
	    }
	    printf("\n");
    }
}