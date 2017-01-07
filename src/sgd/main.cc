// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "graph_io.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("Connected Component with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);

	int num_users = 0;
	ScoreT *h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));
	for (int i = 0; i < nnz; i++) h_rating[i] = (ScoreT)h_weight[i];
	SGDSolver(m, num_users, nnz, h_row_offsets, h_column_indices, h_rating);
	SGDVerifier(m, h_row_offsets, h_column_indices);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	return 0;
}
