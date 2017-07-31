// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "topo_base"
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void initialize(int m, int source, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		expanded[id] = false;
		if(id == source) visited[id] = true;
		else visited[id] = false;
	}
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, bool *changed, bool *visited, bool *expanded, int *frontier_size) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && visited[src] && !expanded[src]) { // visited but not expanded
		expanded[src] = true;
		//atomicAdd(frontier_size, 1);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			DistT new_dist = dist[src] + 1;
			if (new_dist < dist[dst]) {
				DistT old_dist = atomicMin(&dist[dst], new_dist);
				if (new_dist < old_dist) {
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

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	//print_device_info(0);
	DistT zero = 0;
	bool one = 1;
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
	CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(&d_visited[source], &one, sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemset(d_expanded, 0, m * sizeof(bool)));
	int *d_frontier_size;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_size, sizeof(int)));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	//int h_frontier_size = 1;
	//initialize <<<nblocks, nthreads>>> (m, source, d_visited, d_expanded);
	//CudaTest("initializing failed");
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, d_changed, d_visited, d_expanded, d_frontier_size);
		bfs_update <<<nblocks, nthreads>>> (m, d_dist, d_visited);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
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
