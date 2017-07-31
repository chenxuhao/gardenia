// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#include "common.h"
#include "cutil_subset.h"

__global__ void vector_add(int n, ValueType *a, ValueType *b, ValueType *c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int id = tid; id < n; id += blockDim.x * gridDim.x)
		c[id] = a[id] + b[id];
}

void run_gpu(int num, ValueType *h_a, ValueType *h_b, ValueType *h_c) {
	ValueType *d_a, *d_b, *d_c;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, num * sizeof(ValueType)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, num * sizeof(ValueType)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_c, num * sizeof(ValueType)));
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, num * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num * sizeof(ValueType), cudaMemcpyHostToDevice));
	int nthreads = BLOCK_SIZE;
	//int nblocks = 1;
	int nblocks = (num - 1) / nthreads + 1;
	vector_add<<<nblocks, nthreads>>>(num, d_a, d_b, d_c);
	CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, num * sizeof(ValueType), cudaMemcpyDeviceToHost));
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(int argc, char *argv[]) {
	int num = 1024 * 1024;
	if(argc == 2) num = atoi(argv[1]);

	ValueType *h_a = (ValueType *)malloc(num * sizeof(ValueType));
	ValueType *h_b = (ValueType *)malloc(num * sizeof(ValueType));
	ValueType *h_c = (ValueType *)malloc(num * sizeof(ValueType));
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
