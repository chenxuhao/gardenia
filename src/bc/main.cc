// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Betweenness Centrality with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int device = 0;
	if (argc > 2) device = atoi(argv[2]);
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);

	ScoreT *h_scores = (ScoreT *)malloc(m * sizeof(ScoreT));
	for (int i = 0; i < m; i++) h_scores[i] = 0;
	BCSolver(m, nnz, h_row_offsets, h_column_indices, h_scores, device);
	BCVerifier(m, h_row_offsets, h_column_indices, 1, h_scores);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	if(h_weight) free(h_weight);
	return 0;
}
