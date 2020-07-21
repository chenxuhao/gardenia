// Copyright (c) 2016, Xuhao Chen
#include <iostream>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>
#include "tc.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

__global__ void ordered_count(int m, IndexT *row_offsets, IndexT *column_indices, int *total) {
	typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int local_total = 0;
	if (u < m) {
		auto begin_u = row_offsets[u];
		auto end_u = row_offsets[u+1]; 
		for (auto off_u = begin_u; off_u < end_u; ++ off_u) {
			int v = column_indices[off_u];
			int begin_v = row_offsets[v];
			int end_v = row_offsets[v+1];
      auto it = begin_u;
			for (auto off_v = begin_v; off_v < end_v; ++ off_v) {
				int w = column_indices[off_v];
				while (column_indices[it] < w && it < end_u) it ++;
				if (it != end_u && column_indices[it] == w) local_total += 1;
			}
		}
	}
	int block_total = BlockReduce(temp_storage).Sum(local_total);
	if(threadIdx.x == 0) atomicAdd(total, block_total);
}

void TCSolver(Graph &g, uint64_t &total) {
	int64_t m = g.num_vertices();
	int64_t nnz = g.num_edges();
	IndexT *h_row_offsets = g.out_rowptr();
	IndexT *h_column_indices = g.out_colidx();
	//print_device_info(0);
	int zero = 0;
	int *d_row_offsets, *d_column_indices;//, *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	int h_total = 0, *d_total;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(int), cudaMemcpyHostToDevice));

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = maximum_residency(ordered_count, nthreads, 0);
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	ordered_count<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_total);
	CudaTest("solving failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [cuda_base] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
	total = (uint64_t)h_total;
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_total));
}

