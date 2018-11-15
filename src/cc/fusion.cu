// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "gbar.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define CC_VARIANT "fusion"

__device__ void hook(int m, int *row_offsets, int *column_indices, CompT *comp, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			int comp_src = comp[src];
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1];
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				//int comp_dst = comp[dst];
				int comp_dst = __ldg(comp+dst);
				if (comp_src == comp_dst) continue;
				int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
				int low_comp = comp_src + (comp_dst - high_comp);
				if (high_comp == comp[high_comp]) {
					*changed = true;
					comp[high_comp] = low_comp;
				}
			}
		}
	}
}

__device__ void shortcut(int m, int *row_offsets, int *column_indices, CompT *comp) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			while (comp[src] != comp[comp[src]]) {
				comp[src] = comp[comp[src]];
			}
		}
	}
}

__global__ void cc_kernel(int m, int *row_offsets, int *column_indices, CompT *comp, bool *changed, GlobalBarrier gb) {
	while (*changed) {
		*changed = false;
		hook(m, row_offsets, column_indices, comp, changed);
		gb.Sync();
		shortcut(m, row_offsets, column_indices, comp);
		gb.Sync();
	}
}

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *h_row_offsets, int *h_column_indices, int *degrees, CompT *h_comp, bool is_directed) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CompT *d_comp;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(CompT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, m * sizeof(CompT), cudaMemcpyHostToDevice));
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

	int iter = 0;
	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(cc_kernel, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, nthreads));
	GlobalBarrierLifetime gb;
	gb.Setup(nblocks);
	printf("Launching CUDA CC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	h_changed = true;
	CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
	cc_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_comp, d_changed, gb);
	CudaTest("solving cc_kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(CompT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

