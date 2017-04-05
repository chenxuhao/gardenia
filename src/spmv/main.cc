// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Sparse Matrix-Vector Multiplication by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	ValueType *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);

	int num_rows = m;
	int num_cols = m;
	ValueType *h_x = (ValueType *)malloc(m * sizeof(ValueType));
	ValueType *h_y = (ValueType *)malloc(m * sizeof(ValueType));
	ValueType *y_host = (ValueType *)malloc(m * sizeof(ValueType));
	for(int i = 0; i < nnz; i++)
		h_weight[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[]
	for(int i = 0; i < num_cols; i++)
		h_x[i] = rand() / (RAND_MAX + 1.0);
	for(int i = 0; i < num_rows; i++) {
		h_y[i] = rand() / (RAND_MAX + 1.0);
		y_host[i] = h_y[i];
	}

	SpmvSolver(m, nnz, h_row_offsets, h_column_indices, h_weight, h_x, h_y);
	SpmvVerifier(m, h_row_offsets, h_column_indices, h_weight, h_x, h_y, y_host);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_x);
	free(h_y);
	return 0;
}
