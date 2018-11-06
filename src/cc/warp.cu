// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define CC_VARIANT "warp"
#include "cc.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <algorithm>

__global__ void push(int m, const IndexT *row_offsets, const IndexT *column_indices, CompT *comp, bool *changed) {
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int src = warp_id; src < m; src += num_warps) {
		// use two threads to fetch row_offsets[src] and row_offsets[src+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[isrc];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[src+1];
		int comp_src = comp[src];
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			//int comp_dst = comp[dst];
			int comp_dst = __ldg(comp+dst);
			if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
				*changed = true;
				comp[comp_dst] = comp_src;
			}
		}
	}
}

__global__ void update(int m, CompT *comp) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

void CCSolver(int m, int nnz, IndexT *h_row_offsets, IndexT *h_column_indices, int *degree, CompT *h_comp) {
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
	const int max_blocks_per_SM = maximum_residency(push, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA CC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
		//printf("iteration=%d\n", iter);
		push<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_comp, d_changed);
		CudaTest("solving kernel push failed");
		update<<<(m - 1) / nthreads + 1, nthreads>>>(m, d_comp);
		CudaTest("solving kernel update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(CompT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

