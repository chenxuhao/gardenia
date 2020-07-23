// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <algorithm>
#define GPU_BLOCKING
#include "blocking.h"
//#define ENABLE_WARP
#define SPMV_VARIANT "tiling"

template<typename T>
__global__ void initialize(int m, T *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

template<typename T>
__device__ __inline__ T ld_glb_cs(const T *addr) {
	T return_value;
	asm("ld.cs.global.f32 %0, [%1];" : "=f"(return_value) : "l"(addr));
	return return_value;
}

template<typename T>
__device__ __inline__ void st_glb_cs(T value, T *addr) {
	asm("st.cs.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__global__ void spmv_base(int m, const IndexT *Ap, const IndexT *Aj, const ValueT *Ax, const ValueT *x, ValueT *y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m) {
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		ValueT sum = y[row];
		for (int offset = row_begin; offset < row_end; offset ++){
			//sum += Ax[offset] * x[Aj[offset]];
			sum += Ax[offset] * __ldg(x+Aj[offset]);
			//sum += ld_glb_cs<ValueT>(Ax+offset) * __ldg(x+Aj[offset]);
		}
		y[row] = sum;
		//st_glb_cs<ValueT>(sum, y+row);
	}
}

__global__ void spmv_warp(int m, const IndexT * Ap, const IndexT * Aj, const ValueT * Ax, const ValueT * x, ValueT *y) {
	__shared__ ValueT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < m; row += num_warps) {
		if (thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];
		ValueT sum = 0;
		for (int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
			sum += Ax[offset] * __ldg(x+Aj[offset]);

		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if (thread_lane == 0) y[row] += sdata[threadIdx.x];
	}
}

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *h_Ap, IndexT *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degrees) { 
	//print_device_info(0);
	column_blocking(m, h_Ap, h_Aj, h_Ax);

	ValueT *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_Ap_blocked(num_subgraphs), d_Aj_blocked(num_subgraphs);
	vector<ValueT *> d_Ax_blocked(num_subgraphs);

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap_blocked[bid], (m + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ap_blocked[bid], rowptr_blocked[bid], (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Aj_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ax_blocked[bid], values_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT), cudaMemcpyHostToDevice));
	}

	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	int max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
	int max_blocks = max_blocks_per_SM * nSM;
	int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		//Timer tt;
		//tt.Start();
#ifdef ENABLE_WARP
		spmv_warp<<<nblocks, nthreads>>>(m, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_y);
#else
		int mblocks = (m - 1) / nthreads + 1;
		spmv_base<<<mblocks, nthreads>>>(m, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_y);
#endif
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
		//tt.Stop();
		//printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
	}
	CudaTest("solving spmv kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

