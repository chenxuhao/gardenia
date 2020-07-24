// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <cub/cub.cuh>
#include "vc.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"

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

__global__ void first_fit(int m, uint64_t *row_offsets, 
                          int *column_indices, 
                          Worklist2 inwl, int *colors) {
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

__global__ void conflict_resolve(int m, uint64_t *row_offsets, 
                                 int *column_indices, 
                                 Worklist2 inwl, 
                                 Worklist2 outwl, 
                                 int *colors) {
	//typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
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
	//outwl.push_1item<BlockScan>(conflicted, vertex, BLOCK_SIZE);
	if(conflicted) outwl.push(vertex);
}

int VCSolver(Graph &g, int *colors) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
 
	int *d_colors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));

	int nitems = m;
	int num_colors = 0, iter = 0;
	Worklist2 inwl(m), outwl(m);
	Worklist2 *inwlptr = &inwl, *outwlptr = &outwl;
	for(int i = 0; i < m; i ++) inwl.h_queue[i] = i;
	inwl.set_index(m);
	CUDA_SAFE_CALL(cudaMemcpy(inwl.d_queue, inwl.h_queue, m * sizeof(int), cudaMemcpyHostToDevice));
	//thrust::sequence(thrust::device, inwl.d_queue, inwl.d_queue + m);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA VC solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	Timer t;
	t.Start();
	while (nitems > 0) {
		iter ++;
		int nblocks = (nitems - 1) / BLOCK_SIZE + 1;
		first_fit<<<nblocks, BLOCK_SIZE>>>(m, d_row_offsets, d_column_indices, *inwlptr, d_colors);
		conflict_resolve<<<nblocks, BLOCK_SIZE>>>(m, d_row_offsets, d_column_indices, *inwlptr, *outwlptr, d_colors);
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
    printf("\truntime[cuda_linear_bitset] = %f ms, num_colors = %d.\n", t.Millisecs(), num_colors);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
	return num_colors;
}

