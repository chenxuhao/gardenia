// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include "color.h"
#include "graph_io.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("Graph coloring with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	// read graph
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);

	int *colors = (int *)calloc(m, sizeof(int));
	ColorSolver(m, nnz, h_row_offsets, h_column_indices, colors);
	ColorVerifier(m, nnz, h_row_offsets, h_column_indices, colors);
	//write_solution("color-out.txt", colors, m);
	free(h_row_offsets);
	free(h_column_indices);
	free(colors);
	return 0;
}
