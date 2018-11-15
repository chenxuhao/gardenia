// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define CC_VARIANT "partition"
#include "cc.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define GPU_SEGMENTING
#include "segmenting.h"
//#define ENABLE_WARP

__device__ __forceinline__ void expandByCta(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, CompT *comp, bool *changed, int *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(id < m)
		size = row_offsets[id+1] - row_offsets[id];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = id;
			processed[id] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		int src = idx_map[sh_vertex];
		int comp_src = comp[src];
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[edge];
				int comp_dst = comp[dst];
				if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
					*changed = true;
					comp[comp_dst] = comp_src;
				}
			}
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, CompT *comp, bool *changed, int *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	if(id < m && !processed[id]) {
		size = row_offsets[id+1] - row_offsets[id];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = id;
			processed[id] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int src = idx_map[winner];
		int comp_src = comp[src];
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[edge];
				int comp_dst = comp[dst];
				if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
					*changed = true;
					comp[comp_dst] = comp_src;
				}
			}
		}
	}
}

__global__ void hook_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, CompT *comp, bool *changed, int *processed) {
	//expandByCta(m, row_offsets, column_indices, idx_map, comp, changed, processed);
	//expandByWarp(m, row_offsets, column_indices, idx_map, comp, changed, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m && !processed[tid]) {
		int src = idx_map[tid];
		int comp_src = comp[src];
		int row_begin = row_offsets[tid];
		int row_end = row_offsets[tid+1]; 
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

__global__ void hook_warp(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, CompT *comp, bool *changed) {
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

__global__ void shortcut(int m, CompT *comp) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *h_row_offsets, IndexT *h_column_indices, int *degrees, CompT *h_comp, bool is_directed) {
	//print_device_info(0);
	segmenting(m, h_row_offsets, h_column_indices, NULL);

	CompT *d_comp;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(CompT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, m * sizeof(CompT), cudaMemcpyHostToDevice));
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs);
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));

	int *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));

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
	const int max_blocks_per_SM = maximum_residency(hook_warp, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
#endif
	printf("Launching CUDA CC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

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
			CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));
			int bblocks = (n_vertices - 1) / nthreads + 1;
			hook_lb<<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_comp, d_changed, d_processed);
#else
			int mblocks = std::min(max_blocks, DIVIDE_INTO(n_vertices, WARPS_PER_BLOCK));
			hook_warp<<<mblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_comp, d_changed);
#endif
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			//tt.Stop();
			//if(iter == 1) printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
		}
		CudaTest("solving kernel hook failed");
		shortcut<<<nblocks, nthreads>>>(m, d_comp);
		CudaTest("solving kernel shortcut failed");
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

