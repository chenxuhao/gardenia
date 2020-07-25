// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>

//#define SHFL
//#define FUSED
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

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

__global__ void pull_step(int m, const uint64_t *row_offsets, 
                          const VertexId *column_indices, 
                          ScoreT *sums, const ScoreT *outgoing_contrib) {
#ifndef SHFL
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
#endif
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int dst = warp_id; dst < m; dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		ScoreT sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			//int src = column_indices[offset];
			int src = __ldg(column_indices+offset);
			//int src = __ldg(column_indices+offset);
			sum += __ldg(outgoing_contrib+src);
		}
#ifndef SHFL
		// store local sum in shared memory,
		// and reduce local sums to global sum
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(thread_lane == 0) sums[dst] += sdata[threadIdx.x];
#else
		sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  8);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  4);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  2);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  1);
		sum = __shfl_sync(0xFFFFFFFF, sum,  0);
		if(thread_lane == 0) sums[dst] = sum;
#endif
	}
}

// pull operation needs incoming neighbor list
__global__ void pull_fused(int m, const uint64_t *row_offsets, 
                           const VertexId *column_indices, 
                           ScoreT *scores, ScoreT *outgoing_contrib, 
                           float *diff, ScoreT base_score) {
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	float error = 0;
	for(int dst = warp_id; dst < m; dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		ScoreT sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int src = column_indices[offset];
			sum += outgoing_contrib[src];
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
			ScoreT old_score = scores[dst];
			ScoreT new_score = base_score + kDamp * sdata[threadIdx.x];
			scores[dst] = new_score;
			error += fabs(new_score - old_score);
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(Graph &g, ScoreT *scores) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.in_rowptr();
  auto h_column_indices = g.in_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

  std::vector<VertexId> degrees(m);
  for (VertexId i = 0; i < m; i ++) degrees[i] = g.get_degree(i);
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, &degrees[0], m * sizeof(int), cudaMemcpyHostToDevice));
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

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
#ifdef FUSED
	const int max_blocks_per_SM = maximum_residency(pull_fused, nthreads, 0);
#else
	const int max_blocks_per_SM = maximum_residency(pull_step, nthreads, 0);
#endif
	const int max_blocks = max_blocks_per_SM * nSM;
	const int mblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
#ifdef FUSED
		pull_fused <<<mblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, base_score);
#else
		pull_step <<<mblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
#endif
		CudaTest("solving kernel pull failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [cuda_pull_warp] = %f ms.\n", t.Millisecs());
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
