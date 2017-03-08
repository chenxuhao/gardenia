// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Breadth-first Search by Xuhao Chen\n");
	int source = 0;
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	} else if (argc> 2) source = atoi(argv[2]);
	printf("Source vertex: %d\n", source);

	// CSR data structures
	int m, nnz;//, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	//read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);
	read_graph(argc, argv, m, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, false, false, false);
	read_graph(argc, argv, m, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, false, true, false);

	// distance array
	DistT *h_dist = (DistT *) malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) h_dist[i] = MYINFINITY;

	BFSSolver(m, nnz, source, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degree, h_dist);
	//BFSSolver(m, nnz, h_row_offsets, h_column_indices, h_degree, h_dist);
	BFSVerifier(m, source, out_row_offsets, out_column_indices, h_dist);
	//write_solution("bfs-out.txt", m, h_dist);

	//free(h_row_offsets);
	//free(h_column_indices);
	//free(h_degree);
	free(in_row_offsets);
	free(in_column_indices);
	free(in_degree);
	free(out_row_offsets);
	free(out_column_indices);
	free(out_degree);
	free(h_weight);
	free(h_dist);
	return 0;
}
