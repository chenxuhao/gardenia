// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "lb"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define DELTA 0.00000001
#define EPSILON 0.01
#define MAX_ITER 19
#define BLKSIZE 128

__global__ void initialize(float *cur_pagerank, float *next_pagerank, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_pagerank[id] = 1.0f / (float)m;
		next_pagerank[id] = 1.0f / (float)m;
	}
}

typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__global__ void update_neighbors(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLKSIZE];
	__shared__ int src_id[BLKSIZE];

	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int id = tid; total_inputs > 0; id += blockDim.x * gridDim.x, total_inputs--) {
		gather_offsets[threadIdx.x] = 0;
		unsigned row_begin = 0, row_end = 0, degree = 0;
		int scratch_offset = 0;
		int total_edges = 0;
		int src = id;
		if (src < m) {
			row_begin = row_offsets[src];
			row_end = row_offsets[src + 1];
			degree = row_end - row_begin;
		}
		BlockScan(temp_storage).ExclusiveSum(degree, scratch_offset, total_edges);
		int done = 0;
		int neighborsdone = 0;
		int neighboroffset = 0;
		while (total_edges > 0) {
			__syncthreads();
			int i;
			for(i = 0; neighborsdone + i < degree && (scratch_offset + i - done) < BLKSIZE; i++) {
				gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
				src_id[i] = src;
			}
			neighborsdone += i;
			scratch_offset += i;
			__syncthreads();
			int dst = 0;
			int edge = gather_offsets[threadIdx.x];
			float value = 0.85 * cur_pagerank[src_id[threadIdx.x]] / (float)degree;
			if(threadIdx.x < total_edges) {
				dst = column_indices[edge];
				//next_pagerank[dst] += value;
				atomicAdd(&next_pagerank[dst], value);
			}
			total_edges -= BLKSIZE;
			done += BLKSIZE;
		}
	}
}

__global__ void self_update(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockReduce<float, 256> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			float delta = abs(next_pagerank[src] - cur_pagerank[src]);
			local_diff += delta;
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = 0.15 / (float)m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree) {
	unsigned zero = 0;
	float *d_diff, h_diff, e = 0.1;
	float *d_cur_pagerank, *d_next_pagerank;
	double starttime, endtime, runtime;
	int iteration = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_cur_pagerank, m * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_pagerank, m * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	initialize <<<nblocks, nthreads>>> (d_cur_pagerank, d_next_pagerank, m);
	CudaTest("initializing failed");
	const size_t max_blocks = maximum_residency(update_neighbors, nthreads, 0);
	//const size_t max_blocks = 5;
	starttime = rtclock();
	do {
		++iteration;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		nblocks = (m - 1) / nthreads + 1;
		update_neighbors <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank);
		self_update <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank, d_diff);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iteration, h_diff);
	} while (h_diff > EPSILON && iteration < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
