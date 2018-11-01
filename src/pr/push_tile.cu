// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "push_tile"
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define GPU_PARTITION
#include "partition.h"

typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void initialize(int m, ScoreT *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

#if 1
__global__ void push_step(int m, IndexT *row_offsets, IndexT *column_indices, IndexT *idx_map, ScoreT *scores, int *degrees, ScoreT *sums, bool *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m) {
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id+1];
		int src = idx_map[id];
		ScoreT value = scores[src] / (ScoreT)degrees[src];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			atomicAdd(&sums[dst], value);
		}
	}
}
#else
__global__ void push_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *scores, ScoreT *sums, bool *processed) {
	//expandByCta(m, row_offsets, column_indices, scores, sums, processed);
	//expandByWarp(m, row_offsets, column_indices, scores, sums, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int src = tid;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int src_idx[BLOCK_SIZE];
	__shared__ ScoreT value[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	src_idx[tx] = 0;
	value[tx] = 0;
	int row_begin = 0, row_end = 0, degree = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (src < m && !processed[src]) {
		row_begin = row_offsets[src];
		row_end = row_offsets[src+1];
		degree = row_end - row_begin;
		if (degree > 0) value[tx] = scores[src] / (ScoreT)degree;
	}
	BlockScan(temp_storage).ExclusiveSum(degree, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	int neighbor_offset = 0;
	while (total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < degree && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			src_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int edge = gather_offsets[tx];
			int dst = column_indices[edge];
			atomicAdd(&sums[dst], value[src_idx[tx]]);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}
#endif

__global__ void l1norm(int m, ScoreT *scores, ScoreT *sums, float *diff, ScoreT base_score) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0;
	if(u < m) {
		ScoreT new_score = base_score + kDamp * sums[u];
		local_diff += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *h_row_offsets, IndexT *h_column_indices, int *h_degrees, ScoreT *h_scores) {
	column_blocking(m, h_row_offsets, h_column_indices, NULL);
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_sums;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	bool *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(bool)));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_rowptr_blocked(num_subgraphs), d_colidx_blocked(num_subgraphs);
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_rowptr_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_colidx_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
	}

	int iter = 0;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, d_sums);
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(bool)));
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			//Timer tt;
			//tt.Start();
			int msub = ms_of_subgraphs[bid];
			//int nnz = nnzs_of_subgraphs[bid];
			int bblocks = (msub - 1) / nthreads + 1;
			push_step<<<bblocks, nthreads>>>(msub, d_rowptr_blocked[bid], d_colidx_blocked[bid], d_idx_map[bid], d_scores, d_degrees, d_sums, d_processed);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			//tt.Stop();
			//if(iter == 1) printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
		}
		CudaTest("solving kernel push_step failed");
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
		CudaTest("solving kernel l1norm failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %lf\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_processed));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
