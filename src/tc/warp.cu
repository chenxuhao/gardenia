// Copyright (c) 2016, Xuhao Chen
#define TC_VARIANT "topo_warp"
#include <iostream>
#include <cub/cub.cuh>
#include "tc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void ordered_count(int m, int *row_offsets, int *column_indices, int *total) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	int local_total = 0;
	for(int src = warp_id; src < m; src += num_warps) {
		// use two threads to fetch row_offsets[src] and row_offsets[src+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[isrc];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[src+1];
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			if (dst > src) break;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst + 1];
			int it = row_begin;
			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				int dst_dst = column_indices[offset_dst];
				if (dst_dst > dst) break;
				while(column_indices[it] < dst_dst) it ++;
				if(column_indices[it] == dst_dst) local_total += 1;
			}
		}
	}
	int block_total = BlockReduce(temp_storage).Sum(local_total);
	if(threadIdx.x == 0) atomicAdd(total, block_total);
}

void TCSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *h_degree, int *h_total) {
	//print_device_info(0);
	int zero = 0;
	int *d_row_offsets, *d_column_indices;//, *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	int *d_total;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(int), cudaMemcpyHostToDevice));

	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(ordered_count, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	ordered_count<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_total);
	CudaTest("solving failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_total));
	//CUDA_SAFE_CALL(cudaFree(d_degree));
}

