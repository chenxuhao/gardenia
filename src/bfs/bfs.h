// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"

/*
GARDENIA Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Xuhao Chen

Will return distance (or parent) array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.

bfs_omp_base: naive OpenMP implementation, one thread per vertex
bfs_omp_beamer: Beamer's OpenMP implementation, one thread per vertex, using the Direction-Optimizing approach
bfs_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
bfs_topo_vector: topology-driven GPU implementation, one warp per vertex using CUDA
bfs_topo_lb: topology-driven GPU implementation, one thread per edge using CUDA
bfs_linear_base: data-driven GPU implementation, one thread per vertex using CUDA
bfs_linear_lb: data-driven GPU implementation, one thread per edge using CUDA
bfs_bu: one thread per vertex using CUDA, using Beamer's bottom-up approach using CUDA
bfs_afree: topology-driven GPU implementation, atomic free, one thread per vertex using CUDA
bfs_fusion: data-driven GPU implementation with kernel fusion, one thread per edge using CUDA
bfs_hybrid: one thread per vertex using CUDA, using Beamer's Direction-Optimizing approach using CUDA
*/

//void BFSSolver(int m, int nnz, int *row_offsets, int *column_indices, int *degree, DistT *dist);
void BFSSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *in_degree,int *out_degree, DistT *dist);
void BFSVerifier(int m, int source, IndexT *row_offsets, IndexT *column_indices, DistT *dist);
//void write_solution(const char *fname, int m, DistT *dist);
