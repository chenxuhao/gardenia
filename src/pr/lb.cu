// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "pull_lb"
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define FUSED 0
typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
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

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, ScoreT *sums, ScoreT *outgoing_contrib) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int dst = id;
	if(dst < m) {
		size = row_offsets[dst+1] - row_offsets[dst];
	}
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = dst;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
			}
		}
	}
}

__global__ void pull_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *sums, ScoreT *outgoing_contrib) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int dst = tid;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int dst_id[BLOCK_SIZE];
	__shared__ ScoreT incoming_total[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	dst_id[tx] = 0;
	incoming_total[tx] = 0.0;
	int row_begin = 0, row_end = 0, degree = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (dst < m) {
		row_begin = row_offsets[dst];
		row_end = row_offsets[dst+1];
		degree = row_end - row_begin;
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
			dst_id[j] = dst;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int edge = gather_offsets[tx];
			int src = column_indices[edge];
			int dst = dst_id[tx];
			atomicAdd(&incoming_total[dst], outgoing_contrib[src]);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	sums[dst] = incoming_total[tx];
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
		pull_step <<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
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
