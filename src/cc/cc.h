// Copyright 2020, MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Connected Components (CC)
Author: Xuhao Chen 

Will return comp array labelling each vertex with a connected component ID

The CC implementation makes use of the Afforest subgraph sampling algorithm [1], 
which restructures and extends the Shiloach-Vishkin algorithm [2].

[1] Michael Sutton, Tal Ben-Nun, and Amnon Barak. "Optimizing Parallel 
    Graph Connectivity Computation via Subgraph Sampling" Symposium on 
    Parallel and Distributed Processing, IPDPS 2018.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

cc_omp_base: one thread per vertex using OpenMP Shiloach-Vishkin
cc_omp_afforest: one thread per vertex using OpenMP Afforest
cc_base: one thread per vertex using CUDA
cc_warp: one warp per vertex using CUDA
*/

void CCSolver(Graph &g, CompT *comp);
void CCVerifier(Graph &g, CompT *comp);
IndexT SampleFrequentElement(int m, IndexT *comp, int64_t num_samples = 1024);
