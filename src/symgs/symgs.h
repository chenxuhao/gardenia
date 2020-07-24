// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"
#include <vector>
/*
GARDENIA Benchmark Suite
Kernel: Symmetric Gauss-Seidel smoother (SymGS)
Author: Xuhao Chen 

Will return a vector x.

This SymGS implements the optimizations from Phillips et al.[2].

[1] Jongsoo Park et al, Efficient Shared-Memory Implementation of 
	High-Performance Conjugate Gradient Benchmark and Its Application 
	to Unstructured Matrices, SC'14

[2] Everett Phillips and Massimiliano Fatica, A CUDA implementation 
	of the High Performance Conjugate Gradient benchmark, 
	International Workshop on Performance Modeling, Benchmarking and 
	Simulation of High Performance Computer Systems, pp 68-84, 2014

symgs_omp : one thread per row (vertex) using OpenMP
symgs_base: one thread per row (vertex) using CUDA
symgs_warp:   one warp per row (vertex) using CUDA
symgs_vector: one vector per row (vertex) using CUDA
*/

void SymGSSolver(Graph &g, int *indices, ValueT *Ax, ValueT *x, ValueT *b, std::vector<int> color_offsets);
void SymGSVerifier(Graph &g, int *indices, ValueT *Ax, ValueT *test_x, ValueT *x_host, ValueT *b, std::vector<int> color_offsets);
