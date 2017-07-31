// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include <algorithm>
#define SPMV_VARIANT "warp"
#include "spmv.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is 
//   used for accessing the x vector.

texture<float,1> tex_x;
void bind_x(const float * x) { CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_x, x)); }
void unbind_x(const float * x) { CUDA_SAFE_CALL(cudaUnbindTexture(tex_x)); }

__global__ void spmv_vector(int num_rows, const int * Ap,  const int * Aj, const ValueType * Ax, const ValueType * x, ValueType * y) {
	__shared__ ValueType sdata[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps) {
		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

		// compute local sum
		ValueType sum = 0;
		for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
			//sum += Ax[offset] * x[Aj[offset]];
			sum += Ax[offset] * tex1Dfetch(tex_x, Aj[offset]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes warp result
		if (thread_lane == 0)
			y[row] += sdata[threadIdx.x];
		//if (thread_lane == 0) printf("thread_id %d, warp_id %d, warp_lane %d, num_warps %d, sdata %f\n", thread_id, warp_id, warp_lane, num_warps, sdata[threadIdx.x]);
	}
}

void SpmvSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, ValueType *h_Ax, ValueType *h_x, ValueType *h_y) { 
	//print_device_info(0);
	int *d_Ap, *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ValueType *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueType) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));

	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(spmv_vector, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(num_rows, WARPS_PER_BLOCK));
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	bind_x(d_x);
	spmv_vector<<<nblocks, nthreads>>>(num_rows, d_Ap, d_Aj, d_Ax, d_x, d_y);   
	CudaTest("solving failed");
	unbind_x(d_x);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueType) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

