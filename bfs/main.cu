// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "variants.h"

int main(int argc, char *argv[]) {
	printf("Breadth-first Search with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL;
	foru *h_weight = NULL;
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else { printf("Unrecognizable input file format\n"); exit(0); }

	int device = 0;
	if (argc > 2) device = atoi(argv[2]);
	assert(device == 0 || device == 1);
	int deviceCount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
	CUDA_SAFE_CALL(cudaSetDevice(device));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	int nSM = deviceProp.multiProcessorCount;
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", 
			deviceCount, device, deviceProp.name, deviceProp.major, deviceProp.minor, nSM, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));

	foru *h_dist = (foru *) malloc(m * sizeof(foru));
	for(int i = 0; i < m; i ++) {
		h_dist[i] = MYINFINITY;
	}
	foru * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(foru), cudaMemcpyHostToDevice));
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	bfs(m, nnz, d_row_offsets, d_column_indices, d_dist, nSM);
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(foru), cudaMemcpyDeviceToHost));
	printf("Verifying...\n");
	unsigned h_nerr = 0;
	verify(m, h_dist, h_row_offsets, h_column_indices, h_weight, &h_nerr);
	printf("\tNumber of errors = %d.\n", h_nerr);
	write_solution("bfs-out.txt", m, h_dist);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	free(h_row_offsets);
	free(h_column_indices);
	free(h_dist);
	return 0;
}
