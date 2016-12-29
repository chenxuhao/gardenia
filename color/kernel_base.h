// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <cub/cub.cuh>
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "worklistc.h"
#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors

typedef cub::BlockScan<int, BLKSIZE> BlockScan;

__global__ void initialize(int *coloring, int m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		coloring[id] = MAXCOLOR;
	}   
}

__global__ void firstFit(int m, int *csrRowPtr, int *csrColInd, Worklist2 inwl, int *coloring) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	bool forbiddenColors[MAXCOLOR+1];
	int vertex;
	if (inwl.pop_id(id, vertex)) {
		int row_begin = csrRowPtr[vertex];
		int row_end = csrRowPtr[vertex + 1];
		for (int j = 0; j < MAXCOLOR; j++)
			forbiddenColors[j] = false;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = csrColInd[offset];
			int color = coloring[neighbor];
			if(color != MAXCOLOR)
				forbiddenColors[color] = true;
		}
		int vertex_color;
		for (vertex_color = 0; vertex_color < MAXCOLOR; vertex_color++) {
			if (!forbiddenColors[vertex_color]) {
				coloring[vertex] = vertex_color;
				break;
			}
		}
		assert(vertex_color < MAXCOLOR);
	}
}

__global__ void conflictResolve(int m, int *csrRowPtr, int *csrColInd, Worklist2 inwl, Worklist2 outwl, int *coloring) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int conflicted = 0;
	int vertex;
	if (inwl.pop_id(id, vertex)) {
		int row_begin = csrRowPtr[vertex];
		int row_end = csrRowPtr[vertex + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = csrColInd[offset];
			if (coloring[vertex] == coloring[neighbor] && vertex < neighbor) {
				conflicted = 1;
				coloring[vertex] = MAXCOLOR;
				break;
			}
		}
	}
	//outwl.push_1item<BlockScan>(conflicted, vertex, BLKSIZE);
	if(conflicted) outwl.push(vertex);
}

void ColorSolver(int m, int nnz, int *csrRowPtr, int *csrColInd, int *coloring) {
	double starttime, endtime;
	double runtime[ITERATIONS];
	int colors[ITERATIONS];
	int iterations[ITERATIONS];
	int *d_csrRowPtr, *d_csrColInd, *d_coloring;
	printf("Graph coloring data-driven Base version\n");
	for(int i = 0; i < m; i ++) {
		coloring[i] = MAXCOLOR;
	}
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_coloring, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	const size_t max_blocks_1 = maximum_residency(firstFit, BLKSIZE, 0); 
	const size_t max_blocks_2 = maximum_residency(conflictResolve, BLKSIZE, 0); 
	printf("max_blocks_1=%d, max_blocks_2=%d\n", max_blocks_1, max_blocks_2);

	for (int i = 0; i < ITERATIONS; i++) {
		Worklist2 inwl(m), outwl(m);
		Worklist2 *inwlptr = &inwl, *outwlptr = &outwl;
		for(int i = 0; i < m; i ++) {
			inwl.wl[i] = i;
		}
		CUDA_SAFE_CALL(cudaMemcpy(inwl.dindex, &m, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_coloring, coloring, m * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(inwl.dwl, inwl.wl, m * sizeof(int), cudaMemcpyHostToDevice));
		iterations[i] = 0;

		starttime = rtclock();
		int nitems = m;
		//thrust::sequence(thrust::device, inwl.dwl, inwl.dwl + m);
		while (nitems > 0) {
			iterations[i] ++;
			int nblocks = (nitems - 1) / BLKSIZE + 1;
			firstFit<<<nblocks, BLKSIZE>>>(m, d_csrRowPtr, d_csrColInd, *inwlptr, d_coloring);
			conflictResolve<<<nblocks, BLKSIZE>>>(m, d_csrRowPtr, d_csrColInd, *inwlptr, *outwlptr, d_coloring);
			nitems = outwlptr->nitems();
			Worklist2 * tmp = inwlptr;
			inwlptr = outwlptr;
			outwlptr = tmp;
			outwlptr->reset();
		}
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
