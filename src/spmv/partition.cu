// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include <algorithm>
#define SPMV_VARIANT "partition"
#include "spmv.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
#define GPU_PARTITION
#include "partition.h"

__global__ void spmv_warp(int num_rows, const int * Ap,  const int * Aj, const ValueT * Ax, const ValueT * x, ValueT *partial_sums) {
	__shared__ ValueT sdata[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps) {
		if (thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];
		ValueT sum = 0;
		for (int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
			sum += Ax[offset] * __ldg(x+Aj[offset]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes warp result
		if (thread_lane == 0) partial_sums[row] += sdata[threadIdx.x];
	}
}

__global__ void merge_cta(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ScoreT** partial_sums, ScoreT *sums) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ ScoreT sdata[RANGE_WIDTH];
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
				ScoreT local_sum = partial_sums[bid][lid];
				sdata[gid%RANGE_WIDTH] += local_sum;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m)
			sums[global_id] = sdata[local_id];
	}
}


void SpmvSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degree) { 
	//print_device_info(0);
	column_blocking(num_rows, h_Ap, h_Aj, h_Ax);

	ValueT *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));

	int num_subgraphs = (num_rows - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (num_rows - 1) / RANGE_WIDTH + 1;
	vector<IndexT *> d_Ap_blocked(num_subgraphs), d_Aj_blocked(num_subgraphs);
	vector<ValueT *> d_Ax_blocked(num_subgraphs);
	IndexT ** d_range_indices = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	ScoreT ** d_partial_sums = (ScoreT**)malloc(num_subgraphs * sizeof(ScoreT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_range_indices[bid], (num_ranges+1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_sums[bid], ms_of_subgraphs[bid] * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ap_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Aj_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ax_blocked[bid], values_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_range_indices[bid], range_indices[bid], (num_ranges+1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(d_partial_sums[bid], 0, ms_of_subgraphs[bid] * sizeof(ScoreT)));
	}

	printf("copy host pointers to device\n");
	IndexT ** d_range_indices_ptr, **d_idx_map_ptr;
	ScoreT ** d_partial_sums_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_range_indices_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_idx_map_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_sums_ptr, num_subgraphs * sizeof(ScoreT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_range_indices_ptr, d_range_indices, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_idx_map_ptr, d_idx_map, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_sums_ptr, d_partial_sums, num_subgraphs * sizeof(ScoreT*), cudaMemcpyHostToDevice));
	bool *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, num_rows * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, num_rows * sizeof(bool)));

	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(num_rows, WARPS_PER_BLOCK));
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		//Timer tt;
		//tt.Start();
		int n_vertices = ms_of_subgraphs[bid];
		int nnz = nnzs_of_subgraphs[bid];
		int bblocks = (n_vertices - 1) / nthreads + 1;
		spmv_warp<<<bblocks, nthreads>>>(n_vertices, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_partial_sums[bid]);
	}
	CudaTest("solving spmv_warp kernel failed");
	merge_cta <<<num_ranges, nthreads>>>(num_rows, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_sums_ptr, d_y);
	CudaTest("solving merge_cta kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

