// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Xuhao Chen

Will count the number of triangles (cliques of size 3)

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors to be sorted.

tc_omp : one thread per vertex using OpenMP
tc_base: one thread per vertex using CUDA
tc_warp: one warp per vertex using CUDA
*/

void TCSolver(Graph &g, uint64_t &total);
void TCVerifier(Graph &g, uint64_t &test_total);
