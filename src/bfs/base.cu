// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "base"
#include "bfs.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

#ifdef TEXTURE
texture <int, 1, cudaReadModeElementType> row_offsets;
texture <int, 1, cudaReadModeElementType> column_indices;
#endif
__global__ void initialize(int m, DistT *dist) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) dist[id] = MYINFINITY;
}

#ifdef TEXTURE
__global__ void bfs_kernel(int m, DistT *dist, Worklist2 in_queue, Worklist2 out_queue) {
#else
__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 in_queue, Worklist2 out_queue) {
#endif
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(in_queue.pop_id(tid, src)) {
#ifdef TEXTURE
		int row_begin = tex1Dfetch(row_offsets, src);
		int row_end = tex1Dfetch(row_offsets, src + 1);
#else
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
#endif
		for (int offset = row_begin; offset < row_end; ++ offset) {
#ifdef TEXTURE
			int dst = tex1Dfetch(column_indices, offset);
#else
			int dst = column_indices[offset];
#endif
			//DistT new_dist = dist[src] + 1;
			if ((dist[dst] == MYINFINITY) && (atomicCAS(&dist[dst], MYINFINITY, dist[src]+1)==MYINFINITY)) {
			//if (dist[dst] == MYINFINITY) {//Not visited
			//	dist[dst] = new_dist;
				assert(out_queue.push(dst));
			}
		}
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	DistT zero = 0;
	int iter = 0;
	Timer t;
	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
#ifdef TEXTURE
	CUDA_SAFE_CALL(cudaBindTexture(0, row_offsets, d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, column_indices, d_column_indices, nnz * sizeof(int)));
#endif
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 queue1(nnz * 2), queue2(nnz * 2);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int nitems = 1;
	t.Start();
	insert<<<1, nthreads>>>(source, *in_frontier);
	nitems = in_frontier->nitems();
	do {
		++ iter;
		nblocks = (nitems - 1) / nthreads + 1;
		printf("iteration=%d, nblocks=%d, nthreads=%d, frontier_size=%d\n", iter, nblocks, nthreads, nitems);
#ifdef TEXTURE
		bfs_kernel <<<nblocks, nthreads>>> (m, d_dist, *in_frontier, *out_frontier);
#else
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, *in_frontier, *out_frontier);
#endif
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
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
