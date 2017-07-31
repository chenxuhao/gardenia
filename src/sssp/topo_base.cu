// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "topo_base"
#include "sssp.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

//Naive CUDA implementation of the Bellman-Ford algorithm for SSSP
__global__ void initialize(int m, int source, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		expanded[id] = false;
		if(id == source) visited[id] = true;
		else visited[id] = false;
	}
}

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
__global__ void bellman_ford(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, bool *changed, bool *visited, bool *expanded, int *num_frontier) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && visited[src] && !expanded[src]) { // visited but not expanded
		expanded[src] = true;
		//atomicAdd(num_frontier, 1);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			DistT old_dist = dist[dst]; 
			DistT new_dist = dist[src] + weight[offset];
			if (new_dist < old_dist) {
				if (atomicMin(&dist[dst], new_dist) > new_dist) {
					if(expanded[dst]) expanded[dst] = false;
					*changed = true;
				}
			}
		}
	}
}

__global__ void update(int m, DistT *dist, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(dist[id] < MYINFINITY && !visited[id])
			visited[id] = true;
	}
}

/**
 * @brief naive topology-driven mapping GPU SSSP entry point.
 *
 * @param[in] m                 Number of vertices
 * @param[in] h_row_offsets     Host pointer of VertexId to the row offsets queue
 * @param[in] h_column_indices  Host pointer of VertexId to the column indices queue
 * @param[in] h_weight          Host pointer of DistT to the edge weight queue
 * @param[out]h_dist            Host pointer of DistT to the distance queue
 */
void SSSPSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist, int delta) {
	//print_device_info(0);
	DistT zero = 0;
	bool one = 1;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT *d_weight;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(DistT), cudaMemcpyHostToDevice));
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
	int *d_num_frontier;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_frontier, sizeof(int)));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	
	int iter = 0;
	//int h_num_frontier = 1;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SSSP solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bellman_ford<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, d_dist, d_changed, d_visited, d_expanded, d_num_frontier);
		update<<<nblocks, nthreads>>>(m, d_dist, d_visited);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(&h_num_frontier, d_num_frontier, sizeof(int), cudaMemcpyDeviceToHost));
		//printf("iteration %d: num_frontier = %d\n", iter, h_num_frontier);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	CUDA_SAFE_CALL(cudaFree(d_num_frontier));
	return;
}
