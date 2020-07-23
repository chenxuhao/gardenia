// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

// CSR SpMV kernels based on a scalar model (one thread per row)
// Straightforward translation of standard CSR SpMV to CUDA
// where each thread computes y[i] += A[i,:] * x 
// (the dot product of the i-th row of A with the x vector)
__global__ void spmv_csr_scalar(int m, const uint64_t* Ap, 
                                const VertexId* Aj, const ValueT * Ax, 
                                const ValueT * x, ValueT * y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m) {
		ValueT sum = y[row];
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		for (int offset = row_begin; offset < row_end; offset ++){
			sum += Ax[offset] * x[Aj[offset]];
		}
		y[row] = sum;
	}
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
	auto h_Ap = g.in_rowptr();
	auto h_Aj = g.in_colidx();	
	//print_device_info(0);
	uint64_t *d_Ap;
  VertexId *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);
	
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	spmv_csr_scalar <<<nblocks, nthreads>>> (m, d_Ap, d_Aj, d_Ax, d_x, d_y);   
	CudaTest("solving spmv_base kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [cuda_base] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

