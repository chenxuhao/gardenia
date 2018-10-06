// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "graph_io.h"
double hub_factor = 3.0;

int main(int argc, char *argv[]) {
	printf("Connected Component by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	if (argc > 2) hub_factor = atof(argv[2]);

	int m, n, nnz;
	IndexT *h_row_offsets = NULL, *h_column_indices = NULL;
	int *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);
#ifdef SIM
	CompT *h_comp = (CompT *)aligned_alloc(PAGE_SIZE, m * sizeof(CompT));
#else
	CompT *h_comp = (CompT *)malloc(m * sizeof(CompT));
#endif
	for (int i = 0; i < m; i++) h_comp[i] = i;

	CCSolver(m, nnz, h_row_offsets, h_column_indices, h_degree, h_comp);
#ifndef SIM
	CCVerifier(m, h_row_offsets, h_column_indices, h_comp);
	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
#endif
	return 0;
}
