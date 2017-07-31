// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define TC_VARIANT "omp_target"
void TCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *degree, int *total) {
	warm_up();
	Timer t;
	t.Start();
	double t1, t2;
	int *h_total = (int *) malloc(sizeof(int));
#pragma omp target data device(0) map(tofrom:h_total[0:1]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz])
{
	t1 = omp_get_wtime();
#pragma omp target device(0)
{
	int total_num = 0;
	#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 64)
	for (int src = 0; src < m; src ++) {
		int row_begin_src = row_offsets[src];
		int row_end_src = row_offsets[src + 1]; 
		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			int dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst + 1];
			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				int dst_dst = column_indices[offset_dst];
				if (dst_dst > dst)
					break;
				while (column_indices[it] < dst_dst)
					it ++;
				if (dst_dst == column_indices[it])
					total_num ++;
			}
		} 
	}
	h_total[0] = total_num;
}
	t2 = omp_get_wtime();
}
	*total = h_total[0];
	t.Stop();
	//printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, 1000*(t2-t1));
	return;
}
