/*
# Copyright (c) 2010-2012 Christopher Brown
#
# This file is part of Medussa.
#
# Medussa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Medussa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Medussa.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#
*/

#include "medussa_matrix.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------- */

/* Allocate a medussa_dmatrix of dimension mat_0, mat_1 and initialize with data from mat.
   If mat is NULL, the matrix contains all zeros.
   If a memory allocation failure occurs the function returns NULL.
*/
medussa_dmatrix* alloc_medussa_dmatrix( int mat_0, int mat_1, double *mat )
{
    int dataLengthBytes = sizeof(double) * mat_0 * mat_1;

    medussa_dmatrix *result = (medussa_dmatrix*)malloc( sizeof(medussa_dmatrix) );
    if( !result )
        return NULL;

    result->mat_0 = mat_0;
    result->mat_1 = mat_1;
    
    result->mat = (double*)malloc( dataLengthBytes );
    if( !result->mat ){
        free( result );
        return NULL;
    }

    if( mat )
        memcpy( result->mat, mat, dataLengthBytes );
    else
        memset( result->mat, 0, dataLengthBytes );

    return result;
}

/* Free a medussa_dmatrix previously allocated with alloc_medussa_dmatrix */
void free_medussa_dmatrix( medussa_dmatrix *mat )
{
    if( !mat ) return;
    free( mat->mat );
    free( mat );
}

/* -------------------------------------------------------------------- */

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
        printf("Matrix multiplication not defined\n");
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        printf("Output matrix has wrong dimensions\n");
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


int dmatrix_add (double *a,  int a_m,  int a_n,
    double *b,  int b_m,  int b_n,
    double *ab, int ab_m, int ab_n)
{
    int N = a_m * a_n;
    int i;

    if ((a_n != b_n) || (a_m != b_m)) {
        // Dimensions of sources don't match
        printf("dmatrix_add: Dimension mismatch a b | a_n:%d b_n:%d a_m:%d b_m:%d\n", a_n, b_n, a_m, b_m );
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        printf("dmatrix_add: Dimension mismatch ab\n");
        return -2;
    }

    for( i=0; i < N; ++i ){
        ab[i] = a[i] + b[i];
    }

    return 0;
}


int dmatrix_subtract (double *a,  int a_m,  int a_n,
    double *b,  int b_m,  int b_n,
    double *ab, int ab_m, int ab_n)
{
    int N = a_m * a_n;
    int i;

    if ((a_n != b_n) || (a_m != b_m)) {
        // Dimensions of sources don't match
        printf("dmatrix_subtract: Dimension mismatch a b | a_n:%d b_n:%d a_m:%d b_m:%d\n", a_n, b_n, a_m, b_m );
        return -1;
    }
    if ((a_m != ab_m) || (ab_n != b_n)) {
        // Output matrix has wrong dimensions
        printf("dmatrix_subtract: Dimension mismatch ab\n");
        return -2;
    }

    for( i=0; i < N; ++i ){
        ab[i] = a[i] - b[i];
    }

    return 0;
}


void dmatrix_scale (double *a,  int a_m,  int a_n,
    double b,
    double *ab, int ab_m, int ab_n)
{
    int N = a_m * a_n;
    int i;

    for( i=0; i < N; ++i ){
        ab[i] = a[i] * b;
    }
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
