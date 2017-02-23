// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Vertex Coloring (VC)
Author: Xuhao Chen 

Will return colors array assigned to each vertex

This VC implementation makes use of the Gebremedhin and Manne [2] algorithm 
with implementation optimizations from Li et al. [1].

[1] P. Li, X. Chen, Z. Quan, J. Fang, H. Su, T. Tang, and C. Yang, “High
	performance parallel graph coloring on gpgpus,” in Proceedings of
	the 30th IPDPS Workshop, pp. 1–10, 2016.

[2] A. H. Gebremedhin and F. Manne, “Scalable parallel graph coloring algorithms,” 
	Concurrency: Practice and Experience, 2000
*/


void VCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors);
void VCVerifier(int m, int *row_offsets, int *column_indices, int *colors);
