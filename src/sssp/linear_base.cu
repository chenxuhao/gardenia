// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
/*
Naive CUDA implementation of the Bellman-Ford algorithm for SSSP

[1] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, “Work-efficient
	parallel gpu methods for single-source shortest paths,” in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349–359, May 2014
*/

/**
 * @brief naive Bellman_Ford SSSP kernel entry point.
 *
 * @param[in] m                 Number of vertices
 * @param[in] d_row_offsets     Device pointer of VertexId to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_weight          Device pointer of DistT to the edge weight queue
 * @param[out]d_dist            Device pointer of DistT to the distance queue
 * @param[in] d_in_queue        Device pointer of VertexId to the incoming frontier queue
 * @param[out]d_out_queue       Device pointer of VertexId to the outgoing frontier queue
 */
__global__ void bellman_ford(int m, const uint64_t *row_offsets, 
                             const VertexId*column_indices, 
                             DistT *weight, DistT *dist, 
                             Worklist2 in_frontier, 
                             Worklist2 out_frontier) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(in_frontier.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			DistT old_dist = dist[dst];
			DistT new_dist = dist[src] + weight[offset];
			if (new_dist < old_dist) {
				if (atomicMin(&dist[dst], new_dist) > new_dist) out_frontier.push(dst);
			}
		}
	}
}

__global__ void insert(int source, Worklist2 in_frontier) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) in_frontier.push(source);
	return;
}

/**
 * @brief naive data-driven mapping GPU SSSP entry point.
 *
 * @param[in] m                 Number of vertices
 * @param[in] h_row_offsets     Host pointer of VertexId to the row offsets queue
 * @param[in] h_column_indices  Host pointer of VertexId to the column indices queue
 * @param[in] h_weight          Host pointer of DistT to the edge weight queue
 * @param[out]h_dist            Host pointer of DistT to the distance queue
 */

void SSSPSolver(Graph &g, int source, DistT *h_weight, DistT *h_dist, int delta) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	DistT zero = 0;
	DistT *d_weight;
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	Worklist2 wl1(m), wl2(m);
	Worklist2 *in_frontier = &wl1, *out_frontier = &wl2;
	int nitems = 1;
	printf("Launching CUDA SSSP solver (block_size = %d) ...\n", nthreads);

	Timer t;
	t.Start();
	insert<<<1, nthreads>>>(source, *in_frontier);
	nitems = in_frontier->nitems();
	do {
		++ iter;
		nblocks = (nitems - 1) / nthreads + 1;
		//printf("iteration %d: frontier_size = %d\n", iter, nitems);
		//in_frontier->display_items();
		bellman_ford<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, d_dist, *in_frontier, *out_frontier);
		CudaTest("solving failed");
		nitems = out_frontier->nitems();
		Worklist2 *tmp = in_frontier;
		in_frontier = out_frontier;
		out_frontier = tmp;
		out_frontier->reset();
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [cuda_linear_base] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}

