// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define CC_VARIANT "partition"
#include "cc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
#define GPU_PARTITION
#include "partition.h"
#define ENABLE_WARP

/*
Gardenia Benchmark Suite
Kernel: Connected Components (CC)
Author: Xuhao Chen

Will return comp array labelling each vertex with a connected component ID
This CC implementation makes use of the Shiloach-Vishkin algorithm
*/
__global__ void push(int m, int *row_offsets, int *column_indices, int *idx_map, CompT *comp, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		int src = idx_map[tid];
		int comp_src = comp[src];
		int row_begin = row_offsets[tid];
		int row_end = row_offsets[tid+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			int comp_dst = comp[dst];
			if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
				*changed = true;
				comp[comp_dst] = comp_src;
			}
		}
	}
}

__global__ void push_warp(int m, int *row_offsets, int *column_indices, int *idx_map, CompT *comp, bool *changed) {
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int i = warp_id; i < m; i += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[i + thread_lane];
		const int row_begin = ptrs[warp_lane][0];
		const int row_end   = ptrs[warp_lane][1];
		int src = idx_map[i];
		int comp_src = comp[src];
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			int comp_dst = comp[dst];
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

void CCSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *degree, CompT *h_comp) {
	//print_device_info(0);
	column_blocking(m, h_row_offsets, h_column_indices, NULL);

	CompT *d_comp;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(CompT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, m * sizeof(CompT), cudaMemcpyHostToDevice));
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs);
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
	}

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
#ifdef ENABLE_WARP
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(push_warp, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
#endif
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
		//printf("iteration=%d\n", iter);
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			//Timer tt;
			//tt.Start();
			int n_vertices = ms_of_subgraphs[bid];
			int nnz = nnzs_of_subgraphs[bid];
#ifndef ENABLE_WARP
			int bblocks = (n_vertices - 1) / nthreads + 1;
			push<<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_comp, d_changed);
#else
			int mblocks = std::min(max_blocks, DIVIDE_INTO(n_vertices, WARPS_PER_BLOCK));
			push_warp<<<mblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_comp, d_changed);
#endif
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			//tt.Stop();
			//if(iter == 1) printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
		}
		CudaTest("solving kernel push failed");
		update<<<nblocks, nthreads>>>(m, d_comp);
		CudaTest("solving kernel update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(CompT) * m, cudaMemcpyDeviceToHost));

	//CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	//CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_comp));
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

