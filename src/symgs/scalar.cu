// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#define SYMGS_VARIANT "scalar"
#include "symgs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] += A[i,:] * x 
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

/*
texture<float,1> tex_x;
void bind_x(const float * x)
{   CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_x, x));   }
void unbind_x(const float * x)
{   CUDA_SAFE_CALL(cudaUnbindTexture(tex_x)); }

template <bool UseCache>
__inline__ __device__ float fetch_x(const int& i, const float * x)
{
    if (UseCache)
        return tex1Dfetch(tex_x, i);
    else
        return x[i];
}
*/
__global__ void symgs_scalar(int num_rows, int * Ap, int * Aj, ValueType * Ax, ValueType * diag, ValueType * x, ValueType * b) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < num_rows) {
		ValueType sum = b[row];
		int row_begin = Ap[row];
		int row_end = Ap[row + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			sum -= Ax[offset] * x[Aj[offset]]; //fetch_x<UseCache>(Aj[jj], x);
		}
		x[row] = sum / diag[row];
	}
}

void SymGSSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, ValueType *h_Ax, ValueType *h_diag, ValueType *h_x, ValueType *h_b) {
	print_device_info(0);
	Timer t;
	int *d_Ap, *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ValueType *d_Ax, *d_diag, *d_x, *d_b;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueType) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diag, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_diag, h_diag, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	int nthreads = 256;
	int nblocks = (num_rows - 1) / nthreads + 1;

	t.Start();
	//bind_x(d_x);
	symgs_scalar <<<nblocks, nthreads>>> (num_rows, d_Ap, d_Aj, d_Ax, d_diag, d_x, d_b);
	//unbind_x(d_x);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, sizeof(ValueType) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_diag));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_b));
}

