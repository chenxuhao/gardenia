// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Triangle Count by Xuhao Chen (only for undirected graphs)\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	int m, n, nnz;
	IndexType *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);
	//read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true, false, true, false, false);

	int h_total = 0;
	TCSolver(m, nnz, h_row_offsets, h_column_indices, h_degree, &h_total);
#ifndef SIM
	TCVerifier(m, h_row_offsets, h_column_indices, h_total);
	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
#endif
	return 0;
}
