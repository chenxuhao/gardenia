// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <stdio.h>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "variants.h"

#ifndef	ITERATIONS
#define	ITERATIONS 1
#endif
#ifndef	BLKSIZE
#define	BLKSIZE 128
#endif

// store colour of all vertex
void write_solution(char *fname, int *coloring, int n) {
	int i;
	FILE *fp;
	fp = fopen(fname, "w");
	for (i = 0; i < n; i++) {
		//fprintf(fp, "%d:%d\n", i, coloring[i]);
		fprintf(fp, "%d\n", coloring[i]);
	}
	fclose(fp);
}

// check if correctly coloured
void verify(int m, int nnz, int *csrRowPtr, int *csrColInd, int *coloring, int *correct) {
	int i, offset, neighbor_j;
	for (i = 0; i < m; i++) {
		for (offset = csrRowPtr[i]; offset < csrRowPtr[i + 1]; offset++) {
			neighbor_j = csrColInd[offset];
			if (coloring[i] == coloring[neighbor_j] && neighbor_j != i) {
				*correct = 0;
				//printf("coloring[%d] = coloring[%d] = %d\n", i, neighbor_j, coloring[i]);
				break;
			}
		}	
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	// read graph
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight);
	print_device_info(argc, argv);

	int *coloring = (int *)calloc(m, sizeof(int));
	color(m, nnz, h_row_offsets, h_column_indices, coloring);
	write_solution("color-out.txt", coloring, m);
	int correct = 1;
	verify(m, nnz, h_row_offsets, h_column_indices, coloring, &correct);
	if (correct)
		printf("correct.\n");
	else
		printf("incorrect.\n");
	return 0;
}
