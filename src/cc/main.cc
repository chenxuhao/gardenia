// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "graph_io.h"
double hub_factor = 3.0;

int main(int argc, char *argv[]) {
	printf("Connected Component by Xuhao Chen\n");
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc > 2) {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("This is a directed graph\n");
		else printf("This is an undirected graph\n");
	}
	if (!is_directed) symmetrize = true;
	if (argc > 3) hub_factor = atof(argv[3]);

	int m, n, nnz;
	WeightT *h_weight = NULL;
	IndexT *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices;
	int *in_degrees, *out_degrees;
	read_graph(argc, argv, m, n, nnz, out_row_offsets, out_column_indices, out_degrees, h_weight, symmetrize, false, false);
	read_graph(argc, argv, m, n, nnz, in_row_offsets, in_column_indices, in_degrees, h_weight, symmetrize, true, false);
	
#ifdef SIM
	CompT *h_comp = (CompT *)aligned_alloc(PAGE_SIZE, m * sizeof(CompT));
#else
	CompT *h_comp = (CompT *)malloc(m * sizeof(CompT));
#endif
	for (int i = 0; i < m; i++) h_comp[i] = i;

	CCSolver(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degrees, h_comp, is_directed);
#ifndef SIM
	CCVerifier(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, h_comp, is_directed);
	free(in_row_offsets);
	free(out_row_offsets);
	free(in_column_indices);
	free(out_column_indices);
	free(in_degrees);
	free(out_degrees);
#endif
	return 0;
}
