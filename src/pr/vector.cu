// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define PR_VARIANT "pull_vector"

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

template <int VECTORS_PER_BLOCK, int THREADS_PER_VECTOR>
__global__ void pull_vector_kernel(int m, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, const __restrict__ ScoreT *outgoing_contrib, ScoreT *sums) {
	__shared__ ScoreT sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]; // padded to avoid reduction ifs
	__shared__ int ptrs[VECTORS_PER_BLOCK][2];

	const int thread_id	  = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR-1);   // thread index within the vector
	const int vector_id   = thread_id   / THREADS_PER_VECTOR;       // global vector index
	const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;       // vector index within the CTA
	const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;          // total number of active vectors

	for(int dst = vector_id; dst < m; dst += num_vectors) {
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_start = ptrs[vector_lane][0];                   //same as: row_start = row_offsets[row];
		const int row_end   = ptrs[vector_lane][1];                   //same as: row_end   = row_offsets[row+1];

		// compute local sum
		ScoreT sum = 0;
		for(int offset = row_start + thread_lane; offset < row_end; offset += THREADS_PER_VECTOR) {
			//int src = column_indices[offset];
			int src = __ldg(column_indices+offset);
			//sum += outgoing_contrib[src];
			sum += __ldg(outgoing_contrib+src);
		}

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum; __syncthreads();
		if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes vector result
		if (thread_lane == 0) sums[dst] += sdata[threadIdx.x];
		//if (thread_lane == 0) st_glb_cs(sdata[threadIdx.x], sums+dst);
	}
}

template <int THREADS_PER_VECTOR>
void pull_vector(int m, int nSM, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, const ScoreT *outgoing_contrib, ScoreT *sums) {
	const int VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	const int max_blocks_per_SM = maximum_residency(pull_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, VECTORS_PER_BLOCK));
	pull_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<nblocks, BLOCK_SIZE>>>(m, row_offsets, column_indices, outgoing_contrib, sums);
	CudaTest("solving failed");
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
	int nblocks = (m - 1) / nthreads + 1;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const ScoreT base_score = (1.0f - kDamp) / m;
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
		int nnz_per_row = nnz / m;
		if (nnz_per_row <=  2)
			pull_vector<2>(m, nSM, d_row_offsets, d_column_indices, d_contrib, d_sums);
		else if (nnz_per_row <=  4)
			pull_vector<4>(m, nSM, d_row_offsets, d_column_indices, d_contrib, d_sums);
		else if (nnz_per_row <=  8)
			pull_vector<8>(m, nSM, d_row_offsets, d_column_indices, d_contrib, d_sums);
		else if (nnz_per_row <= 16)
			pull_vector<16>(m, nSM, d_row_offsets, d_column_indices, d_contrib, d_sums);
		else
			pull_vector<32>(m, nSM, d_row_offsets, d_column_indices, d_contrib, d_sums);
		CudaTest("solving kernel pull failed");
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
		CudaTest("solving kernel l1norm failed");
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
