// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#include "graph_io.h"
#include "variants.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("Single Source Shortest Path (SSSP) with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}

	// CSR data structures
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);
	int *d_row_offsets, *d_column_indices;
	W_TYPE *d_weight;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(W_TYPE)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(W_TYPE), cudaMemcpyHostToDevice));

	// distance array
	DistT *h_dist = (DistT *) malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) {
		h_dist[i] = MYINFINITY;
	}
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));

	SSSPSolver(m, nnz, d_row_offsets, d_column_indices, d_weight, d_dist);
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	SSSPVerifier(m, h_row_offsets, h_column_indices, h_weight, h_dist);
	//write_solution("sssp-out.txt", m, h_dist);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	free(h_row_offsets);
	free(h_column_indices);
	free(h_weight);
	free(h_dist);
	if(h_degree) free(h_degree);
	return 0;
}
