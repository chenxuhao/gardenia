// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define GPU_BLOCKING
#include "blocking.h"
#define PR_VARIANT "tiling"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
}

__device__ __forceinline__ void expandByCta(int m, const IndexT *row_offsets, const IndexT *column_indices, ScoreT *sums, const ScoreT *outgoing_contrib, bool *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	int dst = id;
	if(dst < m) size = row_offsets[dst+1] - row_offsets[dst];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = dst;
			processed[dst] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		ScoreT sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				//int src = column_indices[edge];
				int src = __ldg(column_indices+edge);
				//sum += outgoing_contrib[src];
				sum += __ldg(outgoing_contrib+src);
			}
		}
		ScoreT block_sum = BlockReduce(temp_storage).Sum(sum);
		if(threadIdx.x == 0) sums[sh_vertex] += block_sum;
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *row_offsets, const IndexT *column_indices, ScoreT *sums, const ScoreT *outgoing_contrib, bool *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];
	owner[warp_id] = -1;
	int size = 0;
	int dst = id;
	if(dst < m && !processed[dst]) {
		size = row_offsets[dst+1] - row_offsets[dst];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = dst;
			processed[dst] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		ScoreT sum = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				//int src = column_indices[edge];
				int src = __ldg(column_indices+edge);
				//sum += outgoing_contrib[src];
				sum += __ldg(outgoing_contrib+src);
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) sums[winner] += sdata[threadIdx.x];
	}
}

__global__ void pull_step(int m, const IndexT *row_offsets, const IndexT *column_indices, ScoreT *sums, const ScoreT *outgoing_contrib, bool *processed) {
	expandByCta(m, row_offsets, column_indices, sums, outgoing_contrib, processed);
	expandByWarp(m, row_offsets, column_indices, sums, outgoing_contrib, processed);
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst < m && !processed[dst]) {
	//if (dst < m) {
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			//IndexT src = column_indices[offset];
			IndexT src = __ldg(column_indices+offset);
			//incoming_total += outgoing_contrib[src];
			incoming_total += __ldg(outgoing_contrib+src);
		}
		if(incoming_total != 0) sums[dst] += incoming_total;
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

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	//print_device_info(0);
/*
	IndexT *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, in_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, in_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
*/
	column_blocking(m, in_row_offsets, in_column_indices, NULL);
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_contrib, *d_scores, *d_sums;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
///*
	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs);
	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets_blocked[bid], (m + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets_blocked[bid], rowptr_blocked[bid], (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
	}
//*/
	bool *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(bool)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	const ScoreT base_score = (1.0f - kDamp) / m;
	initialize<<<nblocks, nthreads>>>(m, d_sums);
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib<<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
		//pull_step<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
///*
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			//Timer tt;
			//tt.Start();
			CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(bool)));
			pull_step<<<nblocks, nthreads>>>(m, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_sums, d_contrib, d_processed);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			//tt.Stop();
			//printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
		}
//*/
		CudaTest("solving kernel pull_step failed");
		l1norm<<<nblocks, nthreads>>>(m, d_scores, d_sums, d_diff, base_score);
		CudaTest("solving kernel l1norm failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}

