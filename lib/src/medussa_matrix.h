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

#ifndef INCLUDED_MEDUSSA_MATRIX_H
#define INCLUDED_MEDUSSA_MATRIX_H

#include <stdio.h>

struct medussa_dmatrix{
    double *mat;
    int mat_0;
    int mat_1;
};
typedef struct medussa_dmatrix medussa_dmatrix;

medussa_dmatrix* alloc_medussa_dmatrix( int mat_0, int mat_1, double *mat ); // pass null data to init the matrix
void free_medussa_dmatrix( medussa_dmatrix *mat );


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

// element-wise add. ab_ij = a_ij + b_ij
int dmatrix_add (double *a,  int a_m,  int a_n,
                double *b,  int b_m,  int b_n,
                double *ab, int ab_m, int ab_n);

// element-wise add. ab_ij = a_ij - b_ij
int dmatrix_subtract (double *a,  int a_m,  int a_n,
                double *b,  int b_m,  int b_n,
                double *ab, int ab_m, int ab_n);

// scalar multiplication. ab_ij = a_ij * b
void dmatrix_scale (double *a,  int a_m,  int a_n,
    double b,
    double *ab, int ab_m, int ab_n);

void fprint_matrix (float  *a, int a_m, int a_n);
void dprint_matrix (double *a, int a_m, int a_n);

#endif /* INCLUDED_MEDUSSA_MATRIX_H */
