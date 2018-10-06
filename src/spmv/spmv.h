// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDENIA Benchmark Suite
Kernel: Sparse Matrix-Vector Multiplication (SpMV)
Author: Xuhao Chen 

Will return vector y

This SpMV implements optimizations from Bell et al.[1] on GPU.

[1] Nathan Bell and Michael Garland, Implementing Sparse Matrix-Vector 
    Multiplication on Throughput-Oriented Processors, SC'09

[2] Samuel Williams et. al, Optimization of Sparse Matrix-Vector 
	Multiplication on Emerging Multicore Platforms, SC'07

[3] Xing Liu et. al, Efficient Sparse Matrix-Vector Multiplication
	on x86-Based Many-Core Processors, ICS’13
    
spmv_omp : one thread per row (vertex) using OpenMP
spmv_base: one thread per row (vertex) using CUDA
spmv_warp: one warp per row (vertex) using CUDA
spmv_vector: one vector per row (vertex) using CUDA
*/

void SpmvSolver(int m, int nnz, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degree);
void SpmvVerifier(int m, int nnz, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, ValueT *test_y);
