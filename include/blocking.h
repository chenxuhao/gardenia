// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// This implements the typical column blocking technique
// for sparse matrix computation

#include "common.h"
#include "timer.h"
#include <vector>

#ifdef GPU_BLOCKING
#define SUBGRAPH_SIZE (1024*256)
#else
#define SUBGRAPH_SIZE (1024*512)
#endif

vector<IndexT *> rowptr_blocked;
vector<IndexT *> colidx_blocked;
vector<ValueT *> values_blocked;
vector<ValueT *> partial_sums;
vector<int> nnzs_of_subgraphs;

// This is for pull model, using incomming edges
void column_blocking(int m, IndexT *rowptr, IndexT *colidx, ValueT *values) {
	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	printf("number of subgraphs and ranges: %d\n", num_subgraphs);

	rowptr_blocked.resize(num_subgraphs);
	colidx_blocked.resize(num_subgraphs);
	values_blocked.resize(num_subgraphs);
	nnzs_of_subgraphs.resize(num_subgraphs);
	partial_sums.resize(num_subgraphs);

	Timer t;
	t.Start();
	for (int i = 0; i < num_subgraphs; ++i) {
		nnzs_of_subgraphs[i] = 0;
	}

	printf("calculating number of edges in each subgraph\n");
	for (IndexT dst = 0; dst < m; ++ dst) {
		int start = rowptr[dst];
		int end = rowptr[dst+1];
		for (IndexT j = start; j < end; ++j) {
			IndexT src = colidx[j];
			int bcol = src / SUBGRAPH_SIZE;
			nnzs_of_subgraphs[bcol]++;
		}
	}

	printf("allocating memory for each subgraph\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		rowptr_blocked[i] = (IndexT *) malloc(sizeof(IndexT) * (m+1));
		colidx_blocked[i] = (IndexT *) malloc(sizeof(IndexT) * nnzs_of_subgraphs[i]);
		if(values != NULL) values_blocked[i] = (ValueT *) malloc(sizeof(ValueT) * nnzs_of_subgraphs[i]);
		partial_sums[i] = (ValueT *) malloc(sizeof(ValueT) * m);
		nnzs_of_subgraphs[i] = 0;
		rowptr_blocked[i][0] = 0;
	}

	printf("constructing the blocked CSR\n");
	for (IndexT dst = 0; dst < m; ++ dst) {
		for (IndexT j = rowptr[dst]; j < rowptr[dst+1]; ++j) {
			IndexT src = colidx[j];
			int bcol = src / SUBGRAPH_SIZE;
			colidx_blocked[bcol][nnzs_of_subgraphs[bcol]] = src;
			if(values != NULL) values_blocked[bcol][nnzs_of_subgraphs[bcol]] = values[j];
			nnzs_of_subgraphs[bcol]++;
		}
		for (int i = 0; i < num_subgraphs; ++i) {
			rowptr_blocked[i][dst+1] = nnzs_of_subgraphs[i];
		}
	}
///*
	printf("printing subgraphs:\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		printf("\tprinting subgraph[%d] (%d edges):\n", i, nnzs_of_subgraphs[i]);
/*
		printf("\trow_offsets: ");
		for (int j = 0; j < m+1; ++j)
			printf("%d ", rowptr_blocked[i][j]);
		printf("\n\tcol_indices: ");
		for (int j = 0; j < nnzs_of_subgraphs[i]; ++j)
			printf("%d ", colidx_blocked[i][j]);
		printf("\n");
//*/
	}
//*/
	t.Stop();
	printf("\truntime [preprocessing] = %f ms.\n", t.Millisecs());
}

void free_partitions() {
	for (size_t i = 0; i < rowptr_blocked.size(); ++i) {
		free(rowptr_blocked[i]);
		free(colidx_blocked[i]);
		free(values_blocked[i]);
	}
}
