// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#define SPMV_VARIANT "cusparse"

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *h_Ap, IndexT *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degrees) { 
	//print_device_info(0);
	int *d_Ap, *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	float *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(float) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(float) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(float) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	const float alpha = 1.0;
	const float beta = 1.0;
	cusparseMatDescr_t descr = NULL;	
	CudaSparseCheck(cusparseCreateMatDescr(&descr));
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cudaStream_t streamId;
	cusparseHandle_t cusparseHandle;
	streamId = NULL;
	cusparseHandle = NULL;
	cudaStreamCreateWithFlags(&streamId, cudaStreamNonBlocking);
	CudaSparseCheck(cusparseCreate(&cusparseHandle));
	CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));

	Timer t;
	t.Start();
	CudaSparseCheck(cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		m, m, nnz, &alpha, descr, d_Ax, d_Ap, d_Aj, d_x, &beta, d_y));
	CudaTest("solving failed");
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	CudaSparseCheck(cusparseDestroy(cusparseHandle));
	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [%s] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", SPMV_VARIANT, time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

