// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "scatter"
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *next_scores, ScoreT base_score) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) next_scores[id] = base_score;
}

__global__ void scatter(int m, int *row_offsets, int *column_indices, ScoreT *cur_scores, ScoreT *next_scores) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		int degree = row_end - row_begin;
		ScoreT value = kDamp * cur_scores[src] / (ScoreT)degree;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			atomicAdd(&next_scores[dst], value);
		}
	}
}

__global__ void reduce(int m, ScoreT *cur_scores, ScoreT *next_scores, float *diff) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0;
	if(src < m) {
		local_diff += fabs(next_scores[src] - cur_scores[src]);
		cur_scores[src] = next_scores[src];
		next_scores[src] = (1.0f - kDamp) / m;
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, ScoreT *h_scores) {
	int *d_row_offsets, *d_column_indices;
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	ScoreT *d_next_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_scores, m * sizeof(ScoreT)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	Timer t;
	int iter = 0;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, d_next_scores, base_score);
	CudaTest("initializing failed");
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		scatter <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_scores, d_next_scores);
		CudaTest("solving kernel scatter failed");
		reduce <<<nblocks, nthreads>>> (m, d_scores, d_next_scores, d_diff);
		CudaTest("solving kernel reduce failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		//printf("iteration=%d, diff=%f\n", iter, h_diff);
		printf(" %2d    %lf\n", iter, h_diff);
		//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_next_scores));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
