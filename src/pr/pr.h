// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#define EPSILON 0.0001
const float kDamp = 0.85;
const float epsilon = 0.0000001;
const float epsilon2 = 0.001;
#ifdef SIM
#define MAX_ITER 3
#else
#define MAX_ITER 30
#endif
/*
GARDENIA Benchmark Suite
Kernel: PageRank (PR)
Author: Xuhao Chen

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.

pr_omp: OpenMP implementation, one thread per vertex
pr_gather: topology-driven GPU implementation using gather/pull approach, one thread per vertex using CUDA
pr_scatter: topology-driven GPU implementation using scatter/push approach, one thread per edge using CUDA
*/

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degree, ScoreT *scores);
void PRVerifier(int m, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degree, ScoreT *scores, double target_error);
