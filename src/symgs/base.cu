// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#define SYMGS_VARIANT "base"
#include "symgs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void gs_kernel(int m, uint64_t * Ap, int * Aj, 
                          int* indices, ValueT * Ax, 
                          ValueT * x, ValueT * b) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m) {
		int inew = indices[id];
		int row_begin = Ap[inew];
		int row_end = Ap[inew+1];
		ValueT rsum = 0;
		ValueT diag = 0;
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			if (inew == j) diag = Ax[jj];
			else rsum += x[j] * Ax[jj];
		}
		if (diag != 0) x[inew] = (b[inew] - rsum) / diag;
	}
}

void gauss_seidel(uint64_t *d_Ap, int *d_Aj, 
                  int *d_indices, ValueT *d_Ax, 
                  ValueT *d_x, ValueT *d_b, 
                  int row_start, int row_stop, int row_step) {
	int m = row_stop - row_start;
	const size_t NUM_BLOCKS = (m - 1) / BLOCK_SIZE + 1;
	//printf("m=%d, nblocks=%ld, nthreads=%ld\n", m, NUM_BLOCKS, BLOCK_SIZE);
	gs_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(m, d_Ap, d_Aj, d_indices+row_start, d_Ax, d_x, d_b);
}

void SymGSSolver(Graph &g, int *h_indices, 
                 ValueT *h_Ax, ValueT *h_x, 
                 ValueT *h_b, std::vector<int> color_offsets) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_Ap = g.in_rowptr();
  auto h_Aj = g.in_colidx();	
  //print_device_info(0);
  uint64_t *d_Ap;
  VertexId *d_Aj;
	int *d_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_indices, m * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_indices, h_indices, m * sizeof(int), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_b;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	printf("Launching CUDA SymGS solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	Timer t;
	t.Start();
	//printf("Forward\n");
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gauss_seidel(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i], color_offsets[i+1], 1);
	//printf("Backward\n");
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gauss_seidel(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i-1], color_offsets[i], 1);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_indices));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_b));
}

