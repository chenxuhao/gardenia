// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "bottom_up"
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void bottom_up_kernel(int m, int *row_offsets, int *column_indices, DistT *depths, bool *changed, bool *front, bool *next, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if(src < m && depths[src] == MYINFINITY) { // not visited
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(front[dst]) { // if the parent is in the current frontier
				//atomicAdd(frontier_size, 1);
				depths[src] = depths[dst] + 1;
				//depths[src] = depth;
				next[src] = true; // put this vertex into the next frontier
				*changed = true;
			}
		}
	}
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *h_degree, DistT *h_depths) {
	print_device_info(0);
	DistT zero = 0;
	bool *d_changed, h_changed;
	bool *front, *next;
	//int *d_num_frontier, h_num_frontier;
	Timer t;
	int iter = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_depths;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, m * sizeof(DistT), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_frontier, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	//h_num_frontier = 1;
	thrust::fill(thrust::device, front + source, front + source + 1, 1); // set the source vertex

	int max_blocks = 6;
	max_blocks = maximum_residency(bottom_up_kernel, nthreads, 0);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bottom_up_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_depths, d_changed, front, next, iter);
		CudaTest("solving failed");
		// swap the queues
		bool *temp = front;
		front = next;
		next = temp;
		thrust::fill(thrust::device, next, next + m, 0);
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(&h_num_frontier, d_num_frontier, sizeof(int), cudaMemcpyDeviceToHost));
		//printf("iteration=%d\n", iter);
		//printf("iteration=%d, num_frontier=%d\n", iter, h_num_frontier);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	//CUDA_SAFE_CALL(cudaFree(d_num_frontier));
	return;
}
