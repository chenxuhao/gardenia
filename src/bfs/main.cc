// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Breadth-first Search (BFS) with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}

	// CSR data structures
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false);

	// distance array
	DistT *h_dist = (DistT *) malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) {
		h_dist[i] = MYINFINITY;
	}

	BFSSolver(m, nnz, h_row_offsets, h_column_indices, h_dist); // start breadth first search
	BFSVerifier(m, h_row_offsets, h_column_indices, h_dist); // verify results
	//write_solution("bfs-out.txt", m, h_dist);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_dist);
	if(h_degree) free(h_degree);
	return 0;
}
