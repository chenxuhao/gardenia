// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#define CC_VARIANT "omp_base"

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *row_offsets, IndexT *column_indices, int *degree, CompT *comp, bool is_directed) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP CC solver (%d threads) ...\n", num_threads);
	#pragma omp parallel for
	for (int n = 0; n < m; n ++) comp[n] = n;
	bool change = true;
	int iter = 0;

	Timer t;
	t.Start();
	while (change) {
		change = false;
		iter++;
		//printf("Executing iteration %d ...\n", iter);
		#pragma omp parallel for schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			CompT comp_src = comp[src];
			IndexT row_begin = row_offsets[src];
			IndexT row_end = row_offsets[src+1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = column_indices[offset];
				CompT comp_dst = comp[dst];
				if (comp_src == comp_dst) continue;
				// Hooking condition so lower component ID wins independent of direction
				int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
				int low_comp = comp_src + (comp_dst - high_comp);
				if (high_comp == comp[high_comp]) {
					change = true;
					comp[high_comp] = low_comp;
				}
			}
		}
		#pragma omp parallel for
		for (int n = 0; n < m; n++) {
			while (comp[n] != comp[comp[n]]) {
				comp[n] = comp[comp[n]];
			}
		}
	}
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	return;
}
