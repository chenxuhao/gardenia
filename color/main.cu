// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <stdio.h>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "variants.h"
#include "verifier.h"

#ifndef	ITERATIONS
#define	ITERATIONS 1
#endif
#ifndef	BLKSIZE
#define	BLKSIZE 128
#endif

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	// read graph
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);
	print_device_info(argc, argv);

	int *coloring = (int *)calloc(m, sizeof(int));
	ColorSolver(m, nnz, h_row_offsets, h_column_indices, coloring);
	//write_solution("color-out.txt", coloring, m);
	ColorVerifier(m, nnz, h_row_offsets, h_column_indices, coloring);
	free(h_row_offsets);
	free(h_column_indices);
	return 0;
}
