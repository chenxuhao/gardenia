// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#define SYMGS_VARIANT "warp"
#include "symgs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void gs_kernel(int num_rows, int * Ap, int * Aj, int* indices, ValueT * Ax, ValueT * x, ValueT * b) {
	__shared__ ValueT sdiags[BLOCK_SIZE/WARP_SIZE];
	__shared__ ValueT sdata[BLOCK_SIZE + 16];                         // padded to avoid reduction conditionals
	__shared__ IndexT ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const IndexT thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
	const IndexT thread_lane = threadIdx.x & (WARP_SIZE - 1);         // thread index within the warp
	const IndexT warp_id	    = thread_id   /  WARP_SIZE;				 // global warp index
	const IndexT warp_lane   = threadIdx.x /  WARP_SIZE;              // warp index within the block
	const IndexT num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps

	for(IndexT index = warp_id; index < num_rows; index += num_warps) {
		if(thread_lane == 0) sdiags[warp_lane] = 0; __syncthreads();
		IndexT row = indices[index];

		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const IndexT row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const IndexT row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

		// initialize local sum
		ValueT sum = 0;

		// accumulate local sums
		for(IndexT jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE) {
			IndexT col = Aj[jj];
			bool diag = row == col;
			sum += diag ? 0 : Ax[jj] * x[col];
			if(diag) sdiags[warp_lane] = Ax[jj];
		}

		// store local sum in shared memory
		sdata[threadIdx.x] = sum; __syncthreads();

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes the result
		if (thread_lane == 0 && sdiags[warp_lane] != 0)
			x[row] = (b[row] - sdata[threadIdx.x]) / sdiags[warp_lane];
	}
}

size_t max_blocks;
void gauss_seidel(int *d_Ap, int *d_Aj, int *d_indices, ValueT *d_Ax, ValueT *d_x, ValueT *d_b, int row_start, int row_stop, int row_step) {
	int num_rows = row_stop - row_start;
	const size_t nblocks = std::min(max_blocks, (size_t)DIVIDE_INTO(num_rows, WARPS_PER_BLOCK));
	//printf("num_rows=%d, nblocks=%ld, nthreads=%d, warp_size=%d\n", num_rows, nblocks, BLOCK_SIZE, WARP_SIZE);
	gs_kernel<<<nblocks, BLOCK_SIZE>>>(num_rows, d_Ap, d_Aj, d_indices+row_start, d_Ax, d_x, d_b);
}

void SymGSSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, int *h_indices, ValueT *h_Ax, ValueT *h_x, ValueT *h_b, std::vector<int> color_offsets) {
	//print_device_info(0);
	int *d_Ap, *d_Aj, *d_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_indices, num_rows * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_indices, h_indices, num_rows * sizeof(int), cudaMemcpyHostToDevice));
	ValueT *d_Ax, *d_x, *d_b;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const size_t nSM = deviceProp.multiProcessorCount;
	const size_t max_blocks_per_SM = maximum_residency(gs_kernel, BLOCK_SIZE, 0);
	max_blocks = max_blocks_per_SM * nSM;
	printf("Launching CUDA SymGS solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	Timer t;
	t.Start();
	//printf("Forward\n");
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gauss_seidel(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i], color_offsets[i+1], 1);
	//printf("Backward\n");
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gauss_seidel(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i-1], color_offsets[i], 1);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, sizeof(ValueT) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_indices));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_b));
}

