// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include <omp.h>
#include "timer.h"
#ifdef SIM
#include "sim.h"
#endif

#define TC_VARIANT "omp_base"
void TCSolver(int m, int nnz, IndexType *row_offsets, IndexType *column_indices, int *degree, int *total) {
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP TC solver (%d threads) ...\n", num_threads);
	Timer t;
	t.Start();
#ifdef SIM
	m5_checkpoint(0,0);
	set_addr_bounds(1,(uint64_t)row_offsets,(uint64_t)&row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)column_indices,(uint64_t)&column_indices[nnz],8);
	printf("Begin of ROI\n");
#endif
	int total_num = 0;
	#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 64)
	for (int src = 0; src < m; src ++) {
		IndexType row_begin_src = row_offsets[src];
		IndexType row_end_src = row_offsets[src + 1]; 
		for (IndexType offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			IndexType dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			IndexType row_begin_dst = row_offsets[dst];
			IndexType row_end_dst = row_offsets[dst + 1];
			for (IndexType offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				IndexType dst_dst = column_indices[offset_dst];
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
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	return;
}
