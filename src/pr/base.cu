// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "gather"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void calc_contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib, bool *active) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		//if (src < m && active[src]) {
		outgoing_contrib[src] = scores[src] / degree[src];
	}
}

// gather operation needs incoming neighbor list
__global__ void gather(int m, int *row_offsets, int *column_indices, ScoreT *scores, ScoreT *contrib, float *diff, bool *active, const ScoreT base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	if (src < m) {
		//if (src < m && active[src]) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		ScoreT incoming_total = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			incoming_total += contrib[dst];
		}
		ScoreT old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *out_row_offsets, int *out_column_indices, int *h_degree, ScoreT *h_scores) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices, *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	ScoreT *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	bool *d_active;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_active, m * sizeof(bool)));

	Timer t;
	int iter = 0;
	int nthreads = BLOCK_SIZE;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		calc_contrib<<<nblocks, nthreads>>>(m, d_scores, d_degree, d_contrib, d_active);
		CudaTest("solving kernel calc_contrib failed");
		gather<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, d_active, base_score);
		CudaTest("solving kernel gather failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		//printf("iteration=%d, diff=%f\n", iter, h_diff);
		printf(" %2d    %lf\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degree));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	//CUDA_SAFE_CALL(cudaFree(d_active));
	return;
}
