// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "symgs.h"
#include "../vc/vc.h"
#include "graph_io.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
	printf("Symmetric Gauss-Seidel smoother by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	int m, n, nnz;
	IndexT *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	ValueT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true, false, true, false, true);

	int num_rows = m;
	int num_cols = m;
	ValueT *h_x = (ValueT *)malloc(m * sizeof(ValueT));
	ValueT *h_b = (ValueT *)malloc(m * sizeof(ValueT));
	ValueT *x_host = (ValueT *)malloc(m * sizeof(ValueT));

	// fill matrix with random values: some matrices have extreme values,
	// which makes correctness testing difficult, especially in single precision
	srand(13);
	//for(int i = 0; i < nnz; i++) h_weight[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[]
	//for(int i = 0; i < nnz; i++) h_weight[i] = rand() / (RAND_MAX + 1.0); // Ax[]
	for(int i = 0; i < num_cols; i++) {
		h_x[i] = rand() / (RAND_MAX + 1.0);
		x_host[i] = h_x[i];
	}
	for(int i = 0; i < num_rows; i++) h_b[i] = rand() / (RAND_MAX + 1.0);

	// identify parallelism using vertex coloring
	int *ordering = (int *)malloc(m * sizeof(int));
	//for(int i = 0; i < num_rows; i++) ordering[i] = i;
	thrust::sequence(ordering, ordering+m);
	int *colors = (int *)malloc(m * sizeof(int));
	for(int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
	int num_colors = VCSolver(m, nnz, h_row_offsets, h_column_indices, colors);
	thrust::sort_by_key(colors, colors+m, ordering);
	int *temp = (int *)malloc((num_colors+1) * sizeof(int));
	thrust::reduce_by_key(colors, colors+m, thrust::constant_iterator<int>(1), thrust::make_discard_iterator(), temp);
	thrust::exclusive_scan(temp, temp+num_colors+1, temp, 0);
	std::vector<int> color_offsets(num_colors+1);
	for(size_t i = 0; i < color_offsets.size(); i ++) color_offsets[i] = temp[i];

	SymGSSolver(m, nnz, h_row_offsets, h_column_indices, ordering, h_weight, h_x, h_b, color_offsets);
	SymGSVerifier(m, h_row_offsets, h_column_indices, ordering, h_weight, h_x, x_host, h_b, color_offsets);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(ordering);
	free(h_x);
	free(h_b);
	return 0;
}
