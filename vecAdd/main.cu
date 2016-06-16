#include <stdio.h>
#include "common.h"
#include "cutil_subset.h"

__global__ void vector_add(int n, int *a, int *b, int *c) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int id = tid; id < n; id += blockDim.x * gridDim.x)
		c[id] = a[id] + b[id];
}

int main(int argc, char *argv[]) {
	int device = 0;
	if(argc>1) atoi(argv[1]);
	assert(device == 0 || device == 1); 
	int deviceCount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
	CUDA_SAFE_CALL(cudaSetDevice(device));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	int nSM = deviceProp.multiProcessorCount;
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", 
			deviceCount, device, deviceProp.name, deviceProp.major, deviceProp.minor, nSM, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
	
	int num = 1024;
	int *h_a = (int *)malloc(num * sizeof(int));
	int *h_b = (int *)malloc(num * sizeof(int));
	int *h_c = (int *)malloc(num * sizeof(int));
	for(int i = 0; i < num; i ++) {
		h_a[i] = 1;
		h_b[i] = 1;
	}
	int *d_a, *d_b, *d_c;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, num * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, num * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_c, num * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, num * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, num * sizeof(int), cudaMemcpyHostToDevice));
	int nthreads = 256;
	int nblocks = 1;
	//int nblocks = num / nthreads;
	vector_add<<<nblocks,nthreads>>>(num, d_a, d_b, d_c);
	CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, num * sizeof(int), cudaMemcpyDeviceToHost));
	for(int i = 0; i < 16; i ++) {
		printf("c[%d]=%d\n", i, h_c[i]);
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}
