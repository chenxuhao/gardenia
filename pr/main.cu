// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "variants.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("PageRank with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);
	print_device_info(argc, argv);

	W_TYPE *d_weight;
	int *d_row_offsets, *d_column_indices, *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(W_TYPE)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	float *h_pr, *d_pr;
	h_pr = (float *) malloc(m * sizeof(float));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pr, m * sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(W_TYPE), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));

	pr(m, nnz, d_row_offsets, d_column_indices, d_degree, d_pr);
	CUDA_SAFE_CALL(cudaMemcpy(h_pr, d_pr, m * sizeof(float), cudaMemcpyDeviceToHost));
	//for(int i = 0; i < 10; i++) printf("pr[%d]=%.8f\n", i, h_pr[i]);
	PRVerifier(m, h_row_offsets, h_column_indices, h_degree, h_pr, EPSILON);

	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_pr));
	free(h_row_offsets);
	free(h_column_indices);
	free(h_weight);
	return 0;
}
