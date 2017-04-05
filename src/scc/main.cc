// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "scc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Strongly Connected Component by Xuhao Chen\n");
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("This is a directed graph\n");
		else printf("This is an undirected graph\n");
	}
	if(!is_directed) symmetrize = true;

	// CSR data structures
	int m, n, nnz;
	WeightT *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, n, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, symmetrize, false, false);
	read_graph(argc, argv, m, n, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, symmetrize, true, false);
	int *scc_root = (int *)malloc(m * sizeof(int));

	SCCSolver(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, scc_root);
	SCCVerifier(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, scc_root);
	
	free(in_row_offsets);
	free(in_column_indices);
	free(in_degree);
	free(out_row_offsets);
	free(out_column_indices);
	free(out_degree);
	free(h_weight);
	free(scc_root);
	return 0;
}
