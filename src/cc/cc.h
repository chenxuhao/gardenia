// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
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
*/

void CCSolver(int m, int nnz, int *row_offsets, int *column_indices, CompT *comp);
void CCVerifier(int m, int *row_offsets, int *column_indices, CompT *comp);
