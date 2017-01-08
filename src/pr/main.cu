// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "graph_io.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("PageRank with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, nnz, out_row_offsets, out_column_indices, out_degree, h_weight);
	read_graph(argc, argv, m, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, false, true);
	#if VARIANT==PR_SCATTER
	h_row_offsets = out_row_offsets; h_column_indices = out_column_indices;
	#else
	h_row_offsets = in_row_offsets; h_column_indices = in_column_indices;
	#endif
	h_degree = out_degree;

	float *h_pr;
	h_pr = (float *) malloc(m * sizeof(float));

	PRSolver(m, nnz, h_row_offsets, h_column_indices, h_degree, h_pr);
	PRVerifier(m, out_row_offsets, out_column_indices, out_degree, h_pr, EPSILON);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_weight);
	return 0;
}
