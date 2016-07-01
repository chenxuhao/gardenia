// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define	MAXCOLOR 128

__global__ void initialize(int *coloring, bool *colored, int m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		coloring[id] = MAXCOLOR;
		colored[id] = false;
	}
}

__global__ void firstFit(int m, int *csrRowPtr, int *csrColInd, int *coloring, bool *changed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	bool forbiddenColors[MAXCOLOR+1];
	if (coloring[id] == MAXCOLOR) {
		for (int i = 0; i < MAXCOLOR; i++)
			forbiddenColors[i] = false;
		int row_begin = csrRowPtr[id];
		int row_end = csrRowPtr[id + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = csrColInd[offset];
			int color = coloring[neighbor];
			forbiddenColors[color] = true;
		}
		int vertex_color;
		for (vertex_color = 0; vertex_color < MAXCOLOR; vertex_color++) {
			if (!forbiddenColors[vertex_color]) {
				coloring[id] = vertex_color;
				break;
			}
		}
		assert(vertex_color < MAXCOLOR);
		*changed = true;
	}
}

__global__ void conflictResolve(int m, int *csrRowPtr, int *csrColInd, int *coloring, bool *colored) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (!colored[id]) {
		int row_begin = csrRowPtr[id];
		int row_end = csrRowPtr[id + 1];
		int offset;
		for (offset = row_begin; offset < row_end; offset ++) {
			int neighbor = csrColInd[offset];
			if (coloring[id] == coloring[neighbor] && id < neighbor) {
				coloring[id] = MAXCOLOR;
				break;
			}
		}
		if(offset == row_end)
			colored[id] = true;
	}
}

void color(int m, int nnz, int *csrRowPtr, int *csrColInd, int *coloring) {
	double starttime, endtime, t1, t2;
	double runtime[ITERATIONS];
	int colors[ITERATIONS];
	int iterations[ITERATIONS];
	double avgtime, avgcolors;
	int *d_csrRowPtr, *d_csrColInd, *d_coloring;
	bool *changed, hchanged;
	bool *d_colored;
	const int blksz = 256;
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_coloring, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colored, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
	const size_t max_blocks_1 = maximum_residency(firstFit, blksz, 0);
	const size_t max_blocks_2 = maximum_residency(conflictResolve, blksz, 0);
	printf("max_blocks_1=%d, max_blocks_2=%d\n", max_blocks_1, max_blocks_2);

	for (int i = 0; i < ITERATIONS; i++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&changed, sizeof(bool)));
		initialize <<<((m - 1) / blksz + 1), blksz>>> (d_coloring, d_colored, m);
		iterations[i] = 0;
		starttime = rtclock();	
		do {
			iterations[i] ++;
			hchanged = false;
			CUDA_SAFE_CALL(cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice));
			int nblocks = (m - 1) / blksz + 1;
			firstFit<<<nblocks, blksz>>>(m, d_csrRowPtr, d_csrColInd, d_coloring, changed);
			conflictResolve<<<nblocks, blksz>>>(m, d_csrRowPtr, d_csrColInd, d_coloring, d_colored);
			CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
		} while (hchanged);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		endtime = rtclock();
		runtime[i] = 1000.0f * (endtime - starttime);
		CUDA_SAFE_CALL(cudaMemcpy(coloring, d_coloring, m * sizeof(int), cudaMemcpyDeviceToHost));
		colors[i] = thrust::reduce(coloring, coloring + m, 0, thrust::maximum<int>()) + 1;
		//colors[i] = thrust::reduce(thrust::device, d_coloring, d_coloring + m, 0, thrust::maximum<int>()) + 1;
	}
	double total_time = 0.0;
	int total_colors = 0;
	int total_iterations = 0;
	for (int i = 0; i < ITERATIONS; i++) {
		total_time += runtime[i];
		total_colors += colors[i];
		total_iterations += iterations[i];
		printf("[%d %.2f %d] ", colors[i], runtime[i], iterations[i]);
	}
	double avg_time = (double)total_time / ITERATIONS;
	double avg_colors = (double)total_colors / ITERATIONS;
	double avg_iterations = (double)total_iterations / ITERATIONS;
	printf("\navg_time %f ms, avg_colors %.2f avg_iterations %.2f\n", avg_time, avg_colors, avg_iterations);
	CUDA_SAFE_CALL(cudaFree(d_csrRowPtr));
	CUDA_SAFE_CALL(cudaFree(d_csrColInd));
	CUDA_SAFE_CALL(cudaFree(d_coloring));
}
