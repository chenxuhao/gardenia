// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#define SYMGS_VARIANT "base"
#include "symgs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

texture<float,1> tex_b;
void bind_b(const float * b) { CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_b, b)); }
void unbind_b(const float * b) { CUDA_SAFE_CALL(cudaUnbindTexture(tex_b)); }

__global__ void gs_kernel(int num_rows, int * Ap, int * Aj, int* indices, ValueT * Ax, ValueT * x, ValueT * b) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num_rows) {
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
		if (diag != 0) x[inew] = (tex1Dfetch(tex_b, inew) - rsum) / diag;
	}
}

void gs_gpu(int *d_Ap, int *d_Aj, int *d_indices, ValueT *d_Ax, ValueT *d_x, ValueT *d_b, int row_start, int row_stop, int row_step) {
	int num_rows = row_stop - row_start;
	const size_t NUM_BLOCKS = (num_rows - 1) / BLOCK_SIZE + 1;
	//printf("num_rows=%d, nblocks=%ld, nthreads=%ld\n", num_rows, NUM_BLOCKS, THREADS_PER_BLOCK);
	gs_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(num_rows, d_Ap, d_Aj, d_indices+row_start, d_Ax, d_x, d_b);
}

void SymGSSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, int *h_indices, ValueT *h_Ax, ValueT *h_x, ValueT *h_b, std::vector<int> color_offsets) {
	//print_device_info(0);
	Timer t;
	int *d_Ap, *d_Aj, *d_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_indices, sizeof(int) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_indices, h_indices, num_rows * sizeof(int), cudaMemcpyHostToDevice));
	ValueT *d_Ax, *d_x, *d_b;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, sizeof(ValueT) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num_rows * sizeof(ValueT), cudaMemcpyHostToDevice));
	printf("Launching CUDA SymGS solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	t.Start();
	bind_b(d_b);
	// Forward
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gs_gpu(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i], color_offsets[i+1], 1);
	// Backward
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gs_gpu(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i-1], color_offsets[i], 1);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	unbind_b(d_b);
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, sizeof(ValueT) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_indices));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_b));
}

