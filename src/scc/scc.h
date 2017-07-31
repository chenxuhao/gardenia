// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#ifndef SCC_H
#define SCC_H
#include "common.h"
/*
GARDENIA Benchmark Suite
Kernel: Strongly Connected Components (SCC)
Author: Xuhao Chen 

Will return comp array labelling each vertex with a strongly connected component ID

This SCC implementation makes use of the FB-Trim [2] algorithm with
implementation optimizations from Hong et al. [1].

[1] S. Hong, N. C. Rodia, and K. Olukotun, “On fast parallel detection
	of strongly connected components (scc) in small-world graphs,” in
	Proceedings of the International Conference on High Performance
	Computing, Networking, Storage and Analysis (SC), pp. 92:1–92:11, 2013.

[2] L. Fleischer, B. Hendrickson, and A. Pinar, “On identifying strongly
	connected components in parallel,” in Proceedings of the 15th IPDPS
	Workshops, pp. 505–511, 2000. 
*/

#define INIT_COLOR 1
// 2^20 = 1048576
#define PIVOT_HASH_CONST 1048575
void SCCSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *scc_root);
void SCCVerifier(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *scc_root);
#endif
