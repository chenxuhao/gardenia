// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "common.h"
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: K-Cliques Listing (KCL)
Author: Xuhao Chen

Will count the number of cliques

Requires input graph:
  - to be directed acyclic graph (DAG)
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

The requirements are done by SquishCSR during graph building.

kcl_base: one thread per vertex using CUDA
kcl_warp: one warp per vertex using CUDA
*/
#define MAX_SIZE 5
typedef unsigned long long AccType; // Accumulator type
void MotifSolver(Graph &g, unsigned k, std::vector<AccType> &acc);
void MotifVerifier(Graph &g, unsigned k, std::vector<AccType> acc);
