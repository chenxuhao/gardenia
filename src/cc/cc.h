// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDENIA Benchmark Suite
Kernel: Connected Components (CC)
Author: Xuhao Chen 

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Shiloach-Vishkin [2] algorithm with
implementation optimizations from Bader et al. [1].

[1] David A Bader, Guojing Cong, and John Feo. "On the architectural
    requirements for efficient execution of graph algorithms." International
    Conference on Parallel Processing, Jul 2005.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

cc_omp : one thread per vertex using OpenMP
cc_base: one thread per vertex using CUDA
cc_warp: one warp per vertex using CUDA
*/

void CCSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, int *degrees, CompT *comp);
void CCVerifier(int m, IndexT *row_offsets, IndexT *column_indices, CompT *comp);
