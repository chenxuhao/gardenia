// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SCC_VARIANT "two-phase"
#include "wcc.h"
#include "cuda_launch_config.hpp"
#include <thrust/reduce.h>
#include "timer.h"
#define debug 0

void SCCSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *h_scc_root) {
	print_device_info(0);
	Timer t;
	int iter = 1;
	int *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_degree, m * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_in_degree, in_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_out_degree, out_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	unsigned *d_colors, *d_locks;
	int *d_scc_root;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, m * sizeof(int)));
	thrust::fill(thrust::device, d_colors, d_colors + m, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + m);

	unsigned char *h_status = (unsigned char*)malloc(m * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, m * sizeof(unsigned char)));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, m * sizeof(unsigned char)));
	bool has_pivot;
	int *d_mark;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mark, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_mark, 0, m * sizeof(int)));
	printf("Start solving SCC detection...\n");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	t.Start();
	first_trim(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, m * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	/*
	if(debug) {
		int num_trimmed = 0;
		for (int i = 0; i < m; i ++) {
			if (is_trimmed(h_status[i]))
				num_trimmed ++;
		}
		printf("%d vertices trimmed in the first trimming\n", num_trimmed);
	}
	//*/
	int source;
	for (int i = 0; i < m; i++) { 
		if(!is_removed(h_status[i])) {
			printf("Vertex %d not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	// phase-1
	printf("Start phase-1...\t");
	has_pivot = false;
	//fwd_reach(m, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	//bwd_reach(m, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
	fwd_reach_lb(m, d_out_row_offsets, d_out_column_indices, d_status, d_scc_root);
	bwd_reach_lb(m, d_in_row_offsets, d_in_column_indices, d_status);
	iterative_trim(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	update_colors(m, d_colors, d_status);
	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * m, cudaMemcpyDeviceToHost));
	find_removed_vertices(m, d_status, d_mark);
	int num_removed = thrust::reduce(thrust::device, d_mark, d_mark + m, 0, thrust::plus<int>());;
	//for (int i = 0; i < m; i++) if(is_removed(h_status[i])) num_removed ++;
	//printf("%d vertices removed in phase-1\n", num_removed);
	/*
	if(debug) {
		int first_scc_size = 0;
		int num_trimmed = 0;
		for (int i = 0; i < m; i++) { 
			if(is_trimmed(h_status[i])) num_trimmed ++;
			else if(is_removed(h_status[i])) first_scc_size ++;
		}
		printf("size of the first scc: %d\n", first_scc_size);
		printf("number of trimmed vertices: %d\n", num_trimmed);
	}
	//*/
	
	if(num_removed != m) {
		printf("Start Trim2...\t\t");
		trim2(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		printf("Done\n");
		unsigned min_color = 6;//thrust::reduce(thrust::device, d_colors, d_colors + m, 0, thrust::maximum<unsigned>());
		printf("Start finding WCC...\t");
		has_pivot = find_wcc(m, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root, min_color);
		printf("Done\n");
		//printf("min_color=%d\n", min_color);

		printf("Start phase-2...\t");
		// phase-2
		while (has_pivot) {
			++ iter;
			has_pivot = false;
			//if(debug) printf("iteration=%d\n", iter);
			fwd_reach(m, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			bwd_reach(m, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
			iterative_trim(m, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
			has_pivot = update(m, d_colors, d_status, d_locks, d_scc_root);
		}
		printf("Done\n");
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(unsigned) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * m, cudaMemcpyDeviceToHost));
	print_statistics(m, h_scc_root, h_status);
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SCC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scc_root));
	CUDA_SAFE_CALL(cudaFree(d_status));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	free(h_status);
}

