// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define VC_VARIANT "topo_warp"
#include "vc.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <algorithm>

__device__ __forceinline__ void assignColor(unsigned *forbiddenColors, int *colors, int node) {
	int i;
	for (i = 0; i < MAXCOLOR/32; i++) {
		int pos = __ffs(forbiddenColors[i]);
		if(pos) {
			colors[node] = i * 32 + pos - 1;
			break;
		}
	}
	assert(i < MAXCOLOR/32);
}

__global__ void first_fit(int m, int *row_offsets, int *column_indices, int *colors, bool *changed) {
///*
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned forbiddenColors[MAXCOLOR/32+1];
	if (colors[id] == MAXCOLOR) {
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id+1];
		for (int j = 0; j < MAXCOLOR/32; j++)
			forbiddenColors[j] = 0xffffffff;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[color / 32] &= ~(1 << (color % 32));
		}
		assignColor(forbiddenColors, colors, id);
		*changed = true;
	}
//*/
/*
	//__shared__ int sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ bool forbiddenColors[BLOCK_SIZE/WARP_SIZE][MAXCOLOR+1];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int src = warp_id; src < m; src += num_warps) {
		if(colors[src] == MAXCOLOR) {
			if(thread_lane < 2)
				ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
			const int row_start = ptrs[warp_lane][0];
			const int row_end   = ptrs[warp_lane][1];

			for (int i = 0; i < MAXCOLOR/WARP_SIZE; i++)
				forbiddenColors[warp_lane][i*WARP_SIZE + thread_lane] = false;
			for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE) {
				int dst = column_indices[offset];
				int color = colors[dst];
				forbiddenColors[warp_lane][color] = true;
			}
			bool valid = false;
			int color_bit = 0;
			int i;
			for (i = 0; i < MAXCOLOR/WARP_SIZE; i++) {
				valid = !forbiddenColors[warp_lane][i*WARP_SIZE + thread_lane];
				color_bit = __ballot(valid);
				if(color_bit) break;
			}
			if (thread_lane == 0) {
				int vertex_color = __ffs(color_bit) + i * WARP_SIZE;
				assert(vertex_color < MAXCOLOR);
				colors[src] = vertex_color;
				*changed = true;
			}
		}
	}
//*/
}

__global__ void conflict_resolve(int m, int *row_offsets, int *column_indices, int *colors, bool *colored) {
/*
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m && !colored[id]) {
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id + 1];
		colored[id] = true;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			if (id < neighbor && colors[id] == colors[neighbor]) {
				colors[id] = MAXCOLOR;
				colored[id] = false;
				break;
			}
		}
	}
*/
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ bool conflicted[BLOCK_SIZE/WARP_SIZE];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int src = warp_id; src < m; src += num_warps) {
		if(!colored[src]) {
			if(thread_lane < 2)
				ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
			const int row_start = ptrs[warp_lane][0];
			const int row_end   = ptrs[warp_lane][1];

			if (thread_lane == 0) conflicted[warp_lane] = false;
			__syncthreads();
			bool is_conflicted = false;
			for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE) {
				int dst = column_indices[offset];
				if(src < dst && colors[src] == colors[dst])
					is_conflicted = true;
				if(__any_sync(0xFFFFFFFF, is_conflicted)) { conflicted[warp_lane] = true; break; }
			}
			if (thread_lane == 0) {
				//if(is_conflicted) {
				if(conflicted[warp_lane]) {
					colors[src] = MAXCOLOR;
					colored[src] = false;
				}
			}
		}
	}
}

int VCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices, *d_colors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));
	bool *d_changed, h_changed, *d_colored;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colored, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_colored, 0, m * sizeof(bool)));

	int num_colors = 0, iter = 0;
	const int nthreads = BLOCK_SIZE;
	const int mblocks = (m - 1) / nthreads + 1;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	//const int nSM = deviceProp.multiProcessorCount;
	//const int max_blocks_per_SM = maximum_residency(conflict_resolve, nthreads, 0);
	//const int max_blocks = max_blocks_per_SM * nSM;
	//const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	const int nblocks = std::min(MAX_BLOCKS, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA VC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();	
	do {
		iter ++;
		//printf("iteration=%d\n", iter);
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		first_fit<<<mblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_changed);
		CudaTest("first_fit failed");
		conflict_resolve<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_colored);
		CudaTest("conflict_resolve failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	CUDA_SAFE_CALL(cudaMemcpy(colors, d_colors, m * sizeof(int), cudaMemcpyDeviceToHost));
	#pragma omp parallel for reduction(max : num_colors)
	for (int n = 0; n < m; n ++)
		num_colors = max(num_colors, colors[n]);
	num_colors ++;	
	printf("\titerations = %d.\n", iter);
	printf("\truntime[%s] = %f ms, num_colors = %d.\n", VC_VARIANT, t.Millisecs(), num_colors);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
	return num_colors;
}
