// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SCC_VARIANT "bitset"
#include "scc.h"
#include "bitset.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "timer.h"
#define debug 0

void SCCSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *h_scc_root) {
	print_device_info(0);
	Timer t;
	int iter = 0;
	int *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	unsigned *d_colors, *d_locks;
	int *d_scc_root;
	unsigned *h_colors = (unsigned *)malloc(m * sizeof(unsigned));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, m * sizeof(int)));
	thrust::fill(thrust::device, d_colors, d_colors + m, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + m);

	unsigned char *h_status = (unsigned char*)malloc(m * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, sizeof(unsigned char) * m));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, m * sizeof(unsigned char)));
	bool has_pivot;
	int source;
	printf("Start solving SCC detection...");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	t.Start();
	first_trim(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, m * sizeof(bool), cudaMemcpyDeviceToHost));
	for (int i = 0; i < m; i++) { 
		if(!is_removed(h_status[i])) {
			printf("vertex %d not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	do {
		++ iter;
		has_pivot = false;
		if(debug) printf("iteration=%d\n", iter);
		fwd_reach(m, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		bwd_reach(m, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
		iterative_trim(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
		has_pivot = update(m, d_colors, d_status, d_locks, d_scc_root);
	} while (has_pivot);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(unsigned) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * m, cudaMemcpyDeviceToHost));
	print_statistics(m, h_scc_root, h_status);
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SCC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	CUDA_SAFE_CALL(cudaFree(d_status));
	free(h_status);
}

