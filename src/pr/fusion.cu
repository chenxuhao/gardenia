// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "fusion"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__device__ void calc_contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
			outgoing_contrib[src] = scores[src] / degree[src];
		}
	}
}

__device__ void pull_step(int m, int *row_offsets, int *column_indices, ScoreT *scores, ScoreT *contrib, float *diff, const ScoreT base_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0.0f;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			ScoreT incoming_total = 0;
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				incoming_total += contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			local_diff += abs(scores[src] - old_score);
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

__global__ void pr_kernel(int m, int *row_offsets, int *column_indices, ScoreT *scores, int *degree, ScoreT *contrib, float *diff, ScoreT base_score, GlobalBarrier gb, int iter) {
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (iter = 0; iter < MAX_ITER; iter ++) {
		calc_contrib(m, scores, degree, contrib);
		gb.Sync();
		*diff = 0;
		pull_step(m, row_offsets, column_indices, scores, contrib, diff, base_score);
		gb.Sync();
		//if(tid==0) printf(" %2d    %lf\n", iter+1, *diff);
		if (*diff < EPSILON) break;
	}
}

void PRSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *out_row_offsets, int *out_column_indices, int *h_degree, ScoreT *h_scores) {
	int *d_row_offsets, *d_column_indices, *d_degree;
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	float *d_diff, h_diff = 0;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
	ScoreT *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = 5;
	max_blocks = maximum_residency(pr_kernel, nthreads, 0);
    cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	nblocks = nSM * max_blocks;
	GlobalBarrierLifetime gb;
	gb.Setup(nblocks);
	printf("max_blocks_per_SM=%d, nSM=%d\n", max_blocks, nSM);
	const ScoreT base_score = (1.0f - kDamp) / m;
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	pr_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_degree, d_contrib, d_diff, base_score, gb, 0);
	CudaTest("solving pr_kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	//printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degree));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
