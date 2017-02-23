// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Sparse Matrix-Vector Multiplication (SpMV)
Author: Xuhao Chen 

Will return vector y

This SpMV implementation makes use of the XXX [2] algorithm with
implementation optimizations from Bell et al. [1].

[1] Nathan Bell and Michael Garland, Implementing Sparse Matrix-Vector 
    Multiplication on Throughput-Oriented Processors, SC'09

[2] Y
    J
*/

#define ValueType float
void SpmvSolver(int m, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y);
void SpmvVerifier(int m, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y, ValueType *y_host);
