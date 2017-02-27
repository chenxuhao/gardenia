// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Breadth-first Search by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	// CSR data structures
	int source = 0;
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, false, false, false);
	read_graph(argc, argv, m, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, false, true, false);
	//read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);

	// distance array
	DistT *h_dist = (DistT *) malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) {
		h_dist[i] = MYINFINITY;
	}

	BFSSolver(m, nnz, source, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degree, h_dist); // start breadth first search
	//BFSSolver(m, nnz, h_row_offsets, h_column_indices, h_degree, h_dist); // start breadth first search
	BFSVerifier(m, source, out_row_offsets, out_column_indices, h_dist); // verify results
	//write_solution("bfs-out.txt", m, h_dist);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_dist);
	if(h_degree) free(h_degree);
	return 0;
}
