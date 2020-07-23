// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define SPMV_VARIANT "warp"

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a warp model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_warp
//   Each row of the CSR matrix is assigned to a warp. The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel. This division of work implies that the
//   CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned). On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work. Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number of
//   elements. This code relies on implicit synchronization among threads
//   in a warp. Note that the texture cache is used for accessing the x vector.

__global__ void spmv_warp(int m, const uint64_t* Ap, 
                          const VertexId* Aj, const ValueT* Ax, 
                          const ValueT* x, ValueT* y) {
	__shared__ ValueT sdata[BLOCK_SIZE + 16];                 // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	sdata[threadIdx.x + 16] = 0.0;
	__syncthreads();

	int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < m; row += num_warps) {
		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

		// compute local sum
		ValueT sum = 0;
		for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
			//sum += Ax[offset] * x[Aj[offset]];
			sum += Ax[offset] * __ldg(x + Aj[offset]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes warp result
		if (thread_lane == 0) y[row] += sdata[threadIdx.x];
	}
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
	auto h_Ap = g.in_rowptr();
	auto h_Aj = g.in_colidx();	
	//print_device_info(0);
	uint64_t *d_Ap;
  VertexId *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);

	int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	int max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
	int max_blocks = max_blocks_per_SM * nSM;
	int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	spmv_warp<<<nblocks, nthreads>>>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);   
	CudaTest("solving spmv_warp failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [cuda_warp] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

