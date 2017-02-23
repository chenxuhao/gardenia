// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#define EPSILON 0.001
#define MAX_ITER 30
/*
GARDINIA Benchmark Suite
Kernel: PageRank (PR)
Author: Xuhao Chen

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.
*/

void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *scores);
void PRVerifier(int m, int *row_offsets, int *column_indices, int *degree, float *scores, double target_error);
