// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "gather_warp"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

__global__ void calc_contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) outgoing_contrib[src] = scores[src] / degree[src];
}

// gather operation needs incoming neighbor list
__global__ void gather(int m, int *row_offsets, int *column_indices, ScoreT *scores, ScoreT *contrib, float *diff, const ScoreT base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	float error = 0;
	for(int src = warp_id; src < m; src += num_warps) {
		// use two threads to fetch row_offsets[src] and row_offsets[src+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[isrc];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[src+1];

		// compute local sum
		ScoreT sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			sum += contrib[dst];
		}
		// store local sum in shared memory
		sdata[threadIdx.x] = sum; __syncthreads();

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		if(thread_lane == 0) {
			ScoreT old_score = scores[src];
			ScoreT new_score = base_score + kDamp * sdata[threadIdx.x];
			scores[src] = new_score;
			error += fabs(new_score - old_score);
		}
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
	float *d_errors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_errors, m * sizeof(float)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int iter = 0;
	const ScoreT base_score = (1.0f - kDamp) / m;
	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(gather, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		calc_contrib<<<(m - 1) / nthreads + 1, nthreads>>>(m, d_scores, d_degree, d_contrib);
		CudaTest("solving kernel calc_contrib failed");
		//CUDA_SAFE_CALL(cudaMemset(d_errors, 0, m * sizeof(float)));
		gather<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, base_score);
		CudaTest("solving kernel gather failed");
		//h_diff = thrust::reduce(thrust::device, d_errors, d_errors+m);
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		//printf("iteration=%d, diff=%f\n", iter, h_diff);
		printf(" %2d    %f\n", iter, h_diff);
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
	return;
}
