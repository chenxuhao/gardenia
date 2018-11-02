// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define VC_VARIANT "linear_warp"
#include <cub/cub.cuh>
#include "vc.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"

__device__ __forceinline__ void assignColor(unsigned int *forbiddenColors, int *colors, int node) {
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

__global__ void first_fit(int *row_offsets, int *column_indices, Worklist2 inwl, int *colors) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned forbiddenColors[MAXCOLOR/32+1];
	int vertex;
	if (inwl.pop_id(id, vertex)) {
		int row_begin = row_offsets[vertex];
		int row_end = row_offsets[vertex + 1];
		for (int j = 0; j < MAXCOLOR/32; j++)
			forbiddenColors[j] = 0xffffffff;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[color / 32] &= ~(1 << (color % 32));
		}
		assignColor(forbiddenColors, colors, vertex);
	}
}

__global__ void conflict_resolve(int nitems, int *row_offsets, int *column_indices, Worklist2 inwl, Worklist2 outwl, int *colors) {
/*
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	int conflicted = 0;
	if (inwl.pop_id(id, vertex)) {
		int row_begin = row_offsets[vertex];
		int row_end = row_offsets[vertex + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			if (colors[vertex] == colors[neighbor] && vertex < neighbor) {
				conflicted = 1;
				colors[vertex] = MAXCOLOR;
				break;
			}
		}
	}
	if(conflicted) outwl.push(vertex);
//*/
///*
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ bool conflicted[BLOCK_SIZE/WARP_SIZE];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int index = warp_id; index < nitems; index += num_warps) {
		int src;
		inwl.pop_id(index, src);
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
		const int row_start = ptrs[warp_lane][0];
		const int row_end   = ptrs[warp_lane][1];
	
		if (thread_lane == 0) conflicted[warp_lane] = false;
		__syncthreads();
		bool is_conflicted = false;
		for(int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			if(src < dst && colors[src] == colors[dst]) is_conflicted = true;
			if(__any_sync(0xFFFFFFFF, is_conflicted)) { conflicted[warp_lane] = true; break; }
		}
		if (thread_lane == 0 && conflicted[warp_lane]) {
			colors[src] = MAXCOLOR;
			outwl.push(src);
		}
	}
	//*/
}

int VCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors) {
	int num_colors = 0, iter = 0;
	int *d_row_offsets, *d_column_indices, *d_colors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));

	int nitems = m;
	Worklist2 inwl(m), outwl(m);
	Worklist2 *inwlptr = &inwl, *outwlptr = &outwl;
	for(int i = 0; i < m; i ++) inwl.h_queue[i] = i;
	inwl.set_index(m);
	CUDA_SAFE_CALL(cudaMemcpy(inwl.d_queue, inwl.h_queue, m * sizeof(int), cudaMemcpyHostToDevice));
	//thrust::sequence(thrust::device, inwl.d_queue, inwl.d_queue + m);
	const int nthreads = BLOCK_SIZE;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(conflict_resolve, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA VC solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	Timer t;
	t.Start();
	while (nitems > 0) {
		iter ++;
		const int mblocks = (nitems - 1) / nthreads + 1;
		first_fit<<<mblocks, nthreads>>>(d_row_offsets, d_column_indices, *inwlptr, d_colors);
		const int nblocks = std::min(max_blocks, DIVIDE_INTO(nitems, WARPS_PER_BLOCK));
		conflict_resolve<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, *inwlptr, *outwlptr, d_colors);
		nitems = outwlptr->nitems();
		Worklist2 * tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;
		outwlptr->reset();
	}
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

