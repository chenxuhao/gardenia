// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sssp.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Single Source Shortest Path by Xuhao Chen\n");
	int source = 0;
	bool is_directed = true;
	bool symmetrize = false;
	int delta = 1;
	if (argc < 2) {
		printf("Usage: %s <graph> [source] [is_directed(0/1)] [delta]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		source = atoi(argv[2]);
		printf("Source vertex: %d\n", source);
		if(argc > 3) {
			is_directed = atoi(argv[3]);
			if(is_directed) printf("This is a directed graph\n");
			else printf("This is an undirected graph\n");
		}
		if(argc > 4) {
			delta = atoi(argv[4]);
			printf("Delta: %d\n", delta);
		}
	}
	if(!is_directed) symmetrize = true;

	// CSR data structures
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	float *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, symmetrize);
	//readMatrix(argv[1], &m, &n, &h_row_offsets, &h_column_indices, &h_weight);

	DistT *h_wt = (DistT *) malloc(nnz * sizeof(DistT));
	DistT *h_dist = (DistT *) malloc(m * sizeof(DistT));
	for(int i = 0; i < nnz; i ++) h_wt[i] = (DistT) h_weight[i];
	for(int i = 0; i < m; i ++) h_dist[i] = kDistInf;

	SSSPSolver(m, nnz, source, h_row_offsets, h_column_indices, h_wt, h_dist, delta);
	SSSPVerifier(m, source, h_row_offsets, h_column_indices, h_wt, h_dist);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_weight);
	free(h_dist);
	if(h_degree) free(h_degree);
	return 0;
}
