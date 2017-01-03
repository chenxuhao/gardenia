// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define COLOR_VARIANT "bitset"
#include <cub/cub.cuh>
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "worklistc.h"
#define	BLKSIZE 128
#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors

typedef cub::BlockScan<int, BLKSIZE> BlockScan;

__global__ void initialize(int m, int *colors) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		colors[id] = MAXCOLOR;
	}   
}

__device__ __forceinline__ void assignColor(unsigned int *forbiddenColors, int *colors, int node) {
	int i;
/*
	int c = 32;
	for (i = 0; i < MAXCOLOR/32; i++) {
		if (forbiddenColors[i] != 0) {
			forbiddenColors[i] &= -(signed)forbiddenColors[i];
			if (forbiddenColors[i]) c--;
			if (forbiddenColors[i] & 0x0000ffff) c -= 16;
	        	if (forbiddenColors[i] & 0x00ff00ff) c -= 8;
        		if (forbiddenColors[i] & 0x0f0f0f0f) c -= 4;
		        if (forbiddenColors[i] & 0x33333333) c -= 2;
        		if (forbiddenColors[i] & 0x55555555) c -= 1;
			colors[node] = c;
			break;
		}
		else
			c += 32;
	}
//*/
///*
	for (i = 0; i < MAXCOLOR/32; i++) {
		int pos = __ffs(forbiddenColors[i]);
		if(pos) {
			colors[node] = i * 32 + pos - 1;
			break;
		}
	}
	assert(i < MAXCOLOR/32);
//*/
}

__global__ void first_fit(int m, int *row_offsets, int *column_indices, Worklist2 inwl, int *colors) {
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

__global__ void conflict_resolve(int m, int *row_offsets, int *column_indices, Worklist2 inwl, Worklist2 outwl, int *colors) {
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
	//outwl.push_1item<BlockScan>(conflicted, vertex, BLKSIZE);
	if(conflicted) outwl.push(vertex);
}

void ColorSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors) {
	double starttime, endtime, runtime;
	int num_colors = 0, iter = 0;
	int *d_row_offsets, *d_column_indices, *d_colors;
	for(int i = 0; i < m; i ++) {
		colors[i] = MAXCOLOR;
	}
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));

	Worklist2 inwl(m), outwl(m);
	Worklist2 *inwlptr = &inwl, *outwlptr = &outwl;
	for(int i = 0; i < m; i ++) {
		inwl.wl[i] = i;
	}
	//initialize <<<((m - 1) / BLKSIZE + 1), BLKSIZE>>> (d_colors, m);

	starttime = rtclock();
	int nitems = m;
	CUDA_SAFE_CALL(cudaMemcpy(inwl.dindex, &m, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(inwl.dwl, inwl.wl, m * sizeof(int), cudaMemcpyHostToDevice));
	//thrust::sequence(thrust::device, inwl.dwl, inwl.dwl + m);
	int iteration = 0;
	while (nitems > 0) {
		iter ++;
		int nblocks = (nitems - 1) / BLKSIZE + 1;
		first_fit<<<nblocks, BLKSIZE>>>(m, d_row_offsets, d_column_indices, *inwlptr, d_colors);
		conflict_resolve<<<nblocks, BLKSIZE>>>(m, d_row_offsets, d_column_indices, *inwlptr, *outwlptr, d_colors);
		nitems = outwlptr->nitems();
		Worklist2 * tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;
		outwlptr->reset();
	}
	cudaDeviceSynchronize();
	endtime = rtclock();
	runtime = 1000.0f * (endtime - starttime);
	CUDA_SAFE_CALL(cudaMemcpy(colors, d_colors, m * sizeof(int), cudaMemcpyDeviceToHost));
	//num_colors = thrust::reduce(thrust::device, d_colors, d_colors + m, 0, thrust::maximum<int>()) + 1;
	num_colors = thrust::reduce(colors, colors + m, 0, thrust::maximum<int>()) + 1;
    printf("\titerations = %d.\n", iter);
    printf("\truntime[%s] = %f ms, num_colors = %d.\n", COLOR_VARIANT, runtime, num_colors);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
}

