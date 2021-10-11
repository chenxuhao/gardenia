// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include "common.h"
#include <graph.hpp>
/*
GARDENIA Benchmark Suite
Kernel: Frequent Subgraph Mining (FSM)
Author: Xuhao Chen

Will count 

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

fsm_omp : one thread per vertex using OpenMP
fsm_base: one thread per vertex using CUDA
fsm_warp: one warp per vertex using CUDA
*/

void FSMSolver(const Graph &g, int minsup, unsigned k, size_t &total);
void FSMVerifier(const Graph &g, int minsup, size_t test_total);
//void FSMSolver(int m, int nnz, int k, IndexT *row_offsets, IndexT *column_indices, ValueT *labels, WeightT *weights, int *degree, long long *total);
//void FSMVerifier(int m, int minsup, IndexT *row_offsets, IndexT *column_indices, ValueT *labels, WeightT *weights, long long test_total);
