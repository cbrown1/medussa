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
