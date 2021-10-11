// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "common.h"
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: 
Author: Xuhao Chen

Will count the number of frequent patterns in an undirected graph 

Requires input graph:
  - to be undirected graph
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

The requirements are done by SquishCSR during graph building.

fsm_base: one thread per vertex using CUDA
fsm_warp: one warp per vertex using CUDA
*/
#define MAX_SIZE 5
#define MAX_NUM_PATTERNS 21251
void FsmSolver(Graph &g, unsigned k, unsigned minsup, int nlabels, int &total_num);
void FsmVerifier(Graph &g, unsigned k, unsigned minsup);
