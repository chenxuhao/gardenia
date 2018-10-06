// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include "common.h"
#include "cutil_subset.h"

__global__ void vector_add(int n, ValueT *a, ValueT *b, ValueT *c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int id = tid; id < n; id += blockDim.x * gridDim.x)
		c[id] = a[id] + b[id];
}

void run_gpu(int num, ValueT *h_a, ValueT *h_b, ValueT *h_c) {
	ValueT *d_a, *d_b, *d_c;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, num * sizeof(ValueT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, num * sizeof(ValueT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_c, num * sizeof(ValueT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, num * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num * sizeof(ValueT), cudaMemcpyHostToDevice));
	int nthreads = BLOCK_SIZE;
	//int nblocks = 1;
	int nblocks = (num - 1) / nthreads + 1;
	vector_add<<<nblocks, nthreads>>>(num, d_a, d_b, d_c);
	CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, num * sizeof(ValueT), cudaMemcpyDeviceToHost));
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(int argc, char *argv[]) {
	int num = 1024 * 1024;
	if(argc == 2) num = atoi(argv[1]);

	ValueT *h_a = (ValueT *)malloc(num * sizeof(ValueT));
	ValueT *h_b = (ValueT *)malloc(num * sizeof(ValueT));
	ValueT *h_c = (ValueT *)malloc(num * sizeof(ValueT));
	for(int i = 0; i < num; i ++) {
		h_a[i] = 1;
		h_b[i] = 1;
	}
	run_gpu(num, h_a, h_b, h_c);
	for(int i = 0; i < 16; i ++) {
		printf("c[%d]=%f\n", i, h_c[i]);
	}
	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}
