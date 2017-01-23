// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("PageRank with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz;//, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, false, false, false);
	read_graph(argc, argv, m, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, false, true, false);

	ScoreT *h_scores = (ScoreT *) malloc(m * sizeof(ScoreT));
	PRSolver(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degree, h_scores);
	PRVerifier(m, out_row_offsets, out_column_indices, out_degree, h_scores, EPSILON);

	free(in_row_offsets);
	free(in_column_indices);
	free(in_degree);
	free(out_row_offsets);
	free(out_column_indices);
	free(out_degree);
	free(h_scores);
	free(h_weight);
	return 0;
}
