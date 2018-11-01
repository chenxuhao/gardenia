// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "graph_io.h"
double hub_factor = 3.0;

int main(int argc, char *argv[]) {
	printf("Sparse Matrix-Vector Multiplication by Xuhao Chen\n");
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("A is not a symmetric matrix\n");
		else printf("A is a symmetric matrix\n");
	}
	if(!is_directed) symmetrize = true;
	if (argc > 3) hub_factor = atof(argv[3]);

	int m, n, nnz;
	IndexT *h_row_offsets = NULL, *h_column_indices = NULL;
	int *h_degree = NULL;
	ValueT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, symmetrize, false, true, false, true);
#ifdef SIM
	if(is_directed) {
		int *in_degree = NULL;
		IndexT *in_row_offsets = NULL, *in_column_indices = NULL;
		ValueT *in_weight = NULL;
		read_graph(argc, argv, m, n, nnz, in_row_offsets, in_column_indices, in_degree, in_weight, false, true, true, false, true);
		h_degree = in_degree;
	}
#endif

#ifdef SIM
	ValueT *h_x = (ValueT *)aligned_alloc(PAGE_SIZE, m * sizeof(ValueT));
#else
	ValueT *h_x = (ValueT *)malloc(m * sizeof(ValueT));
#endif
	ValueT *h_y = (ValueT *)malloc(m * sizeof(ValueT));
	ValueT *y_host = (ValueT *)malloc(m * sizeof(ValueT));
	srand(13);
	for(int i = 0; i < nnz; i++) h_weight[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[] (-1 ~ 1)
	//for(int i = 0; i < nnz; i++) h_weight[i] = rand() / (RAND_MAX + 1.0); // Ax[] (0 ~ 1)
	//for(int i = 0; i < m; i++) h_x[i] = rand() / (RAND_MAX + 1.0);
	for(int i = 0; i < m; i++) h_x[i] = 1.0;
	for(int i = 0; i < m; i++) {
		h_y[i] = 0.0;//rand() / (RAND_MAX + 1.0);
		y_host[i] = h_y[i];
	}

	SpmvSolver(m, nnz, h_row_offsets, h_column_indices, h_weight, h_x, h_y, h_degree);
#ifndef SIM
	SpmvVerifier(m, nnz, h_row_offsets, h_column_indices, h_weight, h_x, y_host, h_y);
	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_x);
	free(h_y);
#endif
	return 0;
}
