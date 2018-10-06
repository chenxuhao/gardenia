// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#define SPMV_VARIANT "cusparse"
#include "spmv.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

void SpmvSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degree) { 
	//print_device_info(0);
	int *d_Ap, *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	float *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(float) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(float) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(float) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, num_rows * sizeof(float), cudaMemcpyHostToDevice));
	int nthreads = BLOCK_SIZE;
	int nblocks = (num_rows - 1) / nthreads + 1;
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
		num_rows, num_rows, nnz, &alpha, descr, d_Ax, d_Ap, d_Aj, d_x, &beta, d_y));
	CudaTest("solving failed");
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	CudaSparseCheck(cusparseDestroy(cusparseHandle));
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

