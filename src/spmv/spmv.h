// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"
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

void SpmvSolver(Graph &g, const ValueT* Ax, const ValueT *x, ValueT *y);
void SpmvVerifier(Graph &g, const ValueT* Ax, const ValueT *x, ValueT *y, ValueT *test_y);
