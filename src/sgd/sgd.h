// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen 

Will return 

This SGD implementation makes use of the XXX [2] algorithm with
implementation optimizations from Bell et al. [1].

[1] N
    M

[2] Y
    J
*/

void SGDSolver(int m, int num_users, int nnz, int *row_offsets, int *column_indices, ScoreT *rating);
void SGDVerifier(int m, int *row_offsets, int *column_indices, ScoreT *test_rating);
