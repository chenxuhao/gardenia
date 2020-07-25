// Copyright 2020 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "csr_graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Xuhao Chen

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies their selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their current distance is less than the min distance for the bin to remove enough 
redundant work that this is faster than removing the vertex from older bins.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest
	path algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, “Work-efficient
	parallel gpu methods for single-source shortest paths,” in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349–359, May 2014

sssp_omp: OpenMP implementation using delta-stepping algorithm, one thread per vertex
sssp_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
sssp_topo_lb: topology-driven GPU implementation, one thread per edge using CUDA
sssp_linear_base: data-driven GPU implementation, one thread per vertex using CUDA
sssp_linear_lb: data-driven GPU implementation, one thread per edge using CUDA
*/

//const DistT kDistInf = numeric_limits<DistT>::max()/2;
#define kDistInf UINT_MAX/2
void SSSPSolver(Graph &g, int source, DistT *weight, DistT *dist, int delta);
void SSSPVerifier(Graph &g, int source, DistT *weight, DistT *dist);

