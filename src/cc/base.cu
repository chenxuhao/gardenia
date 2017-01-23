// Copyright (c) 2016, Xuhao Chen
#define CC_VARIANT "topology"
#include "cc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

/*
Gardenia Benchmark Suite
Kernel: Connected Components (CC)
Author: Xuhao Chen

Will return comp array labelling each vertex with a connected component ID
This CC implementation makes use of the Shiloach-Vishkin algorithm
*/

__global__ void initialize(int m, CompT *comp) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		comp[id] = id;
	}
}

__global__ void cc_kernel1(int m, int *row_offsets, int *column_indices, CompT *comp, bool *changed) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			int comp_src = comp[src];
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1]; 
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				int comp_dst = comp[dst];
				if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
					*changed = true;
					comp[comp_dst] = comp_src;
				}
			}
		}
	}
}

__global__ void cc_kernel2(int m, int *row_offsets, int *column_indices, CompT *comp) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			while (comp[src] != comp[comp[src]]) {
				comp[src] = comp[comp[src]];
			}
		}
	}
}

void CCSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, CompT *h_comp) {
	print_device_info(0);
	bool h_changed, *d_changed;
	Timer t;
	int iter = 0;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	CompT *d_comp;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(CompT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = maximum_residency(cc_kernel1, nthreads, 0);
	initialize <<<nblocks, nthreads>>> (m, d_comp);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
		printf("iteration=%d\n", iter);
		cc_kernel1<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_comp, d_changed);
		CudaTest("solving kernel1 failed");
		cc_kernel2<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_comp);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(CompT) * m, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	//CUDA_SAFE_CALL(cudaFree(d_degree));
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

