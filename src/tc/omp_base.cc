// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include <omp.h>
#include "timer.h"

#define TC_VARIANT "omp_base"
void TCSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, int *degree, int *total) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP TC solver (%d threads) ...\n", num_threads);
	Timer t;
	t.Start();
	int total_num = 0;
	#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 64)
	for (int src = 0; src < m; src ++) {
		IndexT row_begin_src = row_offsets[src];
		IndexT row_end_src = row_offsets[src + 1]; 
		for (IndexT offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			IndexT dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			IndexT row_begin_dst = row_offsets[dst];
			IndexT row_end_dst = row_offsets[dst + 1];
			for (IndexT offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				IndexT dst_dst = column_indices[offset_dst];
				if (dst_dst > dst)
					break;
				while (column_indices[it] < dst_dst)
					it ++;
				if (dst_dst == column_indices[it])
					total_num ++;
			}
		} 
	}
	*total = total_num;
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	return;
}
