// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include "vc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Vertex Coloring by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	// read graph
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);

	int *colors = (int *)calloc(m, sizeof(int));
	for(int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
	VCSolver(m, nnz, h_row_offsets, h_column_indices, colors);
	VCVerifier(m, h_row_offsets, h_column_indices, colors);
	//write_solution(m, "color-out.txt", colors);
	free(h_row_offsets);
	free(h_column_indices);
	free(colors);
	return 0;
}
