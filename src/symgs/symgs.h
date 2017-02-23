// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#define ValueType float
/*
GARDINIA Benchmark Suite
Kernel: Symmetric Gauss-Seidel smoother (SymGS)
Author: Xuhao Chen 

Will return 

This SymGS implementation makes use of the XXX [2] algorithm with
implementation optimizations from Bell et al. [1].

[1] N
    M

[2] Y
    J
*/

void SymGSSolver(int m, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *diag, ValueType *x, ValueType *b);
void SymGSVerifier(int num_rows, int *Ap, int *Aj, ValueType *Ax, ValueType *diag, ValueType *test_x, ValueType *x_host, ValueType *b);
