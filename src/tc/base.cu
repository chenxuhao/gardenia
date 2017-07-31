// Copyright (c) 2016, Xuhao Chen
#define TC_VARIANT "topo_base"
#include <iostream>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>
#include "tc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void ordered_count(int m, int *row_offsets, int *column_indices, int *total) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int local_total = 0;
	if (src < m) {
		int row_begin_src = row_offsets[src];
		int row_end_src = row_offsets[src + 1]; 
		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			int dst = column_indices[offset_src];
			if (dst > src) break;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst + 1];
			int it = row_begin_src;
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

// heuristic to see if sufficently dense power-law graph
bool WorthRelabelling(int m, int nnz, int *row_offsets, int *column_indices, int *degree) {
	int64_t average_degree = nnz / m;
	if (average_degree < 10)
		return false;
	unsigned sp = 1;
	int64_t num_samples = min(int64_t(1000), int64_t(m));
	int64_t sample_total = 0;
	vector<int64_t> samples(num_samples);
	for (int64_t trial=0; trial < num_samples; trial++) {
		samples[trial] = degree[sp++];
		sample_total += samples[trial];
	}
	sort(samples.begin(), samples.end());
	double sample_average = static_cast<double>(sample_total) / num_samples;
	double sample_median = samples[num_samples/2];
	return sample_average / 2 > sample_median;
}

// uses heuristic to see if worth relabeling
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

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = maximum_residency(ordered_count, nthreads, 0);
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	if (WorthRelabelling(m, nnz, h_row_offsets, h_column_indices, h_degree))
		printf("worth relabelling\n");
	//	ordered_count<<<nblocks, nthreads>>>(m, row_offsets, column_indices, total);
	//else
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

