// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Symmetric Gauss-Seidel smoother (SymGS)
Author: Xuhao Chen 

Will return a vector x.

This SymGS implements the algorithm with implementation 
optimizations from Park et al. [1].

[1] Jongsoo Park et al, Efficient Shared-Memory Implementation of 
	High-Performance Conjugate Gradient Benchmark and Its Application 
	to Unstructured Matrices, SC'14
*/

void SymGSSolver(int m, int nnz, int *Ap, int *Aj, int *indices, ValueType *Ax, ValueType *x, ValueType *b, std::vector<int> color_offsets);
void SymGSVerifier(int num_rows, int *Ap, int *Aj, int *indices, ValueType *Ax, ValueType *test_x, ValueType *x_host, ValueType *b, std::vector<int> color_offsets);
