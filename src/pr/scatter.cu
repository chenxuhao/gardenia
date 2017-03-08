// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "scatter"
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *cur_scores, ScoreT *next_scores, ScoreT init_score, ScoreT base_score) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_scores[id] = init_score;
		next_scores[id] = base_score;
	}
}

__global__ void scatter(int m, int *row_offsets, int *column_indices, ScoreT *cur_scores, ScoreT *next_scores) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
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
}

__global__ void reduce(int m, ScoreT *cur_scores, ScoreT *next_scores, float *diff) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			local_diff += fabs(next_scores[src] - cur_scores[src]);
			cur_scores[src] = next_scores[src];
			next_scores[src] = (1.0f - kDamp) / m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, ScoreT *h_scores) {
	float *d_diff, h_diff;
	Timer t;
	ScoreT *d_next_scores;
	int iter = 0;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices;
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	const ScoreT init_score = 1.0f / m;
	const ScoreT base_score = (1.0f - kDamp) / m;
	printf("base_score=%.8f, init_score=%.8f\n", base_score, init_score);
	initialize <<<nblocks, nthreads>>> (m, d_scores, d_next_scores, init_score, base_score);
	CudaTest("initializing failed");
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));

	int max_blocks = maximum_residency(scatter, nthreads, 0);
	//const size_t max_blocks = 5;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		scatter <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_scores, d_next_scores);
		CudaTest("solving kernel1 failed");
		reduce <<<nblocks, nthreads>>> (m, d_scores, d_next_scores, d_diff);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iter, h_diff);
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
