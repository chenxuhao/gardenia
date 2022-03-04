// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include "common.h"
/*
GARDENIA Benchmark Suite
Kernel: Frequent Subgraph Mining
Author: Xuhao Chen

Will count 

Requires input graph:
  - to be undirected
  - no duplicate edges
  - neighborhoods are sorted by vertex identifiers

fsm_omp : one thread per embedding using OpenMP
fsm_base: one thread per embedding using CUDA
fsm_warp: one warp per embedding using CUDA
*/

#define DEBUG 0
void FSMSolver(int m, int nnz, int k, int minsup, IndexT *row_offsets, IndexT *column_indices, int *labels, long long *total);
//void FSMVerifier(int m, int k, IndexT *row_offsets, IndexT *column_indices, long long test_total);
