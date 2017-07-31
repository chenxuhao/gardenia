// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define VC_VARIANT "topo_base"
#include "vc.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void first_fit(int m, int *row_offsets, int *column_indices, int *colors, bool *changed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	bool forbiddenColors[MAXCOLOR+1];
	if (colors[id] == MAXCOLOR) {
		for (int i = 0; i < MAXCOLOR; i++)
			forbiddenColors[i] = false;
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[color] = true;
		}
		int vertex_color;
		for (vertex_color = 0; vertex_color < MAXCOLOR; vertex_color++) {
			if (!forbiddenColors[vertex_color]) {
				colors[id] = vertex_color;
				break;
			}
		}
		assert(vertex_color < MAXCOLOR);
		*changed = true;
	}
}

__global__ void conflict_resolve(int m, int *row_offsets, int *column_indices, int *colors, bool *colored) {
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
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA VC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();	
	do {
		iter ++;
		//printf("iteration=%d\n", iter);
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		first_fit<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_changed);
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
