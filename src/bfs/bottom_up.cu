// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define BFS_VARIANT "bottom_up"

__global__ void bottom_up_kernel(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *front, int *next, DistT *depths, bool *changed, int depth) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if(dst < m && depths[dst] == MYINFINITY) { // not visited
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT src = column_indices[offset];
			if(__ldg(front+src)) { // if the parent is in the current frontier
			//if(front[src]) {
				depths[dst] = depth;
				next[dst] = 1; // put this vertex into the next frontier
				*changed = true;
				break;
			}
		}
	}
}

__global__ void insert(int source, int *front) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) front[source] = 1;
}

void BFSSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *in_degrees, int *out_degrees, DistT *h_depths) {
	//print_device_info(0);
	DistT zero = 0;
	IndexT *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_depths;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	bool *d_changed, h_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	//int *d_num_frontier;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_frontier, sizeof(int)));
	int *front, *next;
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	//int h_num_frontier = 1;
	insert<<<1, 1>>>(source, front); // set the source vertex
	printf("Launching CUDA BFS solver (%d threads/CTA, %d blocks) ...\n", nthreads, nblocks);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bottom_up_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, front, next, d_depths, d_changed, iter);
		CudaTest("solving failed");
		// swap the queues
		int *temp = front;
		front = next;
		next = temp;
		CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(&h_num_frontier, d_num_frontier, sizeof(int), cudaMemcpyDeviceToHost));
		//printf("iteration=%d, num_frontier=%d\n", iter, h_num_frontier);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(front));
	CUDA_SAFE_CALL(cudaFree(next));
	return;
}
