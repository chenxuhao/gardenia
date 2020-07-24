// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"

/*
GARDENIA Benchmark Suite
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

vc_omp: OpenMP implementation, one thread per vertex
vc_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
vc_topo_bitset: topology-driven GPU implementation using bitset operations, one thread per vertex using CUDA
vc_linear_base: data-driven GPU implementation, one thread per vertex using CUDA
vc_linear_bitset: data-driven GPU implementation using bitset operations, one thread per vertex using CUDA
*/

int VCSolver(Graph &g, int *colors);
void VCVerifier(Graph &g, int *colors);

