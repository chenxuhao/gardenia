// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "topo_vector"
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
#include <algorithm>

__global__ void initialize(int m, int source, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		expanded[id] = false;
		if(id == source) visited[id] = true;
		else visited[id] = false;
	}
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, bool *changed, bool *visited, bool *expanded, int *frontier_size, int depth) {
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int src = warp_id; src < m; src += num_warps) {
		if(visited[src] && !expanded[src]) { // visited but not expanded
			expanded[src] = true;
			// use two threads to fetch Ap[row] and Ap[row+1]
			// this is considerably faster than the straightforward version
			if(thread_lane < 2)
				ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
			const int row_begin = ptrs[warp_lane][0];                   //same as: row_start = row_offsets[row];
			const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[row+1];
			for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
				int dst = column_indices[offset];
				//if (dist[dst] > depth) {
				if (dist[dst] == MYINFINITY) {
					//DistT old_dist = atomicMin(&dist[dst], new_dist);
					//if (new_dist < old_dist) {
					dist[dst] = depth;
					*changed = true;
				}
			}
		}
	}
}

__global__ void bfs_update(int m, DistT *dist, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(dist[id] < MYINFINITY && !visited[id])
			visited[id] = true;
	}
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *in_degree, int *h_degree, DistT *h_dist) {
	//print_device_info(0);
	DistT zero = 0;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	bool *d_changed, h_changed, *d_visited, *d_expanded;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_expanded, 0, m * sizeof(bool)));
	int *d_frontier_size;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_size, sizeof(int)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int mblocks = (m - 1) / nthreads + 1;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(bfs_kernel, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	//int h_frontier_size = 1;
	initialize <<<mblocks, nthreads>>> (m, source, d_visited, d_expanded);
	CudaTest("initializing failed");
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, d_changed, d_visited, d_expanded, d_frontier_size, iter);
		bfs_update <<<mblocks, nthreads>>> (m, d_dist, d_visited);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//printf("iteration %d: frontier_size = %d\n", iter, h_frontier_size);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	CUDA_SAFE_CALL(cudaFree(d_frontier_size));
	return;
}
