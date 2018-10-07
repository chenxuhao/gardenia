// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "pull"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define FUSED 0
typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
}

__global__ void pull_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *sums, ScoreT *outgoing_contrib) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst < m) {
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT src = column_indices[offset];
			incoming_total += outgoing_contrib[src];
		}
		sums[dst] = incoming_total;
	}
}

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

// pull operation needs incoming neighbor list
__global__ void pull_fused(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *scores, ScoreT *outgoing_contrib, float *diff, ScoreT base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	if (src < m) {
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src + 1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT dst = column_indices[offset];
			incoming_total += outgoing_contrib[dst];
		}
		ScoreT old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	//print_device_info(0);
	IndexT *d_row_offsets, *d_column_indices;
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, in_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, in_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores, *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib<<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
#if FUSED
		pull_fused <<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, base_score);
#else
		pull_step <<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
#endif
		CudaTest("solving kernel pull failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
