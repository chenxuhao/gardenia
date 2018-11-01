// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include <algorithm>
#define SPMV_VARIANT "partition"
#include "spmv.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
#define GPU_SEGMENTING
#include "segmenting.h"
//#define ENABLE_WARP

template<typename T>
__global__ void initialize(int m, T *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

template<typename T>
__device__ __inline__ T ld_glb_cs(const T *addr) {
	T return_value;
	//asm("ld.cs.global.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	asm("ld.cs.global.f32 %0, [%1];" : "=f"(return_value) : "l"(addr));
	return return_value;
}

template<typename T>
__device__ __inline__ void st_glb_cs(T value, T *addr) {
	asm("st.cs.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__global__ void spmv_base(int m, const IndexT * Ap, const IndexT * Aj, const ValueT * Ax, const ValueT * x, ValueT * partial_sums) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m) {
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		ValueT sum = 0;
		for (int offset = row_begin; offset < row_end; offset ++){
			//sum += Ax[offset] * x[Aj[offset]];
			sum += Ax[offset] * __ldg(x+Aj[offset]);
			//sum += ld_glb_cs<ValueT>(Ax+offset) * __ldg(x+Aj[offset]);
		}
		//partial_sums[row] = sum;
		st_glb_cs<ValueT>(sum, partial_sums+row);
	}
}

__global__ void spmv_warp(int m, const IndexT * Ap, const IndexT * Aj, const ValueT * Ax, const ValueT * x, ValueT *partial_sums) {
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
		if (thread_lane == 0) partial_sums[row] += sdata[threadIdx.x];
	}
}

__global__ void merge_cta(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ValueT** partial_sums, ValueT *y) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ ValueT sdata[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		sdata[tx + i] = 0;
	}
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				ValueT local_sum = partial_sums[bid][lid];
				sdata[gid%RANGE_WIDTH] += local_sum;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m)
			y[global_id] += sdata[local_id];
	}
}

void SpmvSolver(int m, int nnz, int *h_Ap, int *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degree) { 
	//print_device_info(0);
	segmenting(m, h_Ap, h_Aj, h_Ax);

	ValueT *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (m - 1) / RANGE_WIDTH + 1;
	vector<IndexT *> d_Ap_blocked(num_subgraphs), d_Aj_blocked(num_subgraphs);
	vector<ValueT *> d_Ax_blocked(num_subgraphs);
	IndexT ** d_range_indices = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	ValueT ** d_partial_sums = (ValueT**)malloc(num_subgraphs * sizeof(ValueT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_range_indices[bid], (num_ranges+1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_sums[bid], ms_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ap_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Aj_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ax_blocked[bid], values_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_range_indices[bid], range_indices[bid], (num_ranges+1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	}

	printf("copy host pointers to device\n");
	IndexT ** d_range_indices_ptr, **d_idx_map_ptr;
	ValueT ** d_partial_sums_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_range_indices_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_idx_map_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_sums_ptr, num_subgraphs * sizeof(ValueT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_range_indices_ptr, d_range_indices, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_idx_map_ptr, d_idx_map, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_sums_ptr, d_partial_sums, num_subgraphs * sizeof(ValueT*), cudaMemcpyHostToDevice));
	bool *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(bool)));

	const int nthreads = BLOCK_SIZE;
	int mblocks = (m - 1) / nthreads + 1;
	initialize<bool> <<<mblocks, nthreads>>> (m, d_processed);
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int msub = ms_of_subgraphs[bid];
		mblocks = (msub - 1) / nthreads + 1;
		initialize<ValueT> <<<mblocks, nthreads>>> (msub, d_partial_sums[bid]);
	}
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
		int msub = ms_of_subgraphs[bid];
		int nnz = nnzs_of_subgraphs[bid];
#ifdef ENABLE_WARP
		nblocks = std::min(max_blocks, DIVIDE_INTO(msub, WARPS_PER_BLOCK));
		spmv_warp<<<nblocks, nthreads>>>(msub, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_partial_sums[bid]);
#else
		int bblocks = (msub - 1) / nthreads + 1;
		spmv_base<<<bblocks, nthreads>>>(msub, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_partial_sums[bid]);
#endif
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
		//tt.Stop();
		//printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
	}
	CudaTest("solving spmv kernel failed");
	merge_cta <<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_sums_ptr, d_y);
	CudaTest("solving merge kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

