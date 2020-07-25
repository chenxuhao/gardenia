// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"
#define EPSILON 0.0001
const float kDamp = 0.85;
const float epsilon = 0.0000001;
const float epsilon2 = 0.001;
#ifdef SIM
#define MAX_ITER 3
#else
#define MAX_ITER 100
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
pr_base: topology-driven GPU implementation using pull approach, one thread per vertex using CUDA
pr_push: topology-driven GPU implementation using push approach, one thread per edge using CUDA
*/

void PRSolver(Graph &g, ScoreT *scores);
void PRVerifier(Graph &g, ScoreT *scores, double target_error);

