// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"
#include "timer.h"

void TCSolver(Graph &g, uint64_t &total) {
	int m = g.num_vertices();
	IndexT *row_offsets = g.out_rowptr();
	IndexT *column_indices = g.out_colidx();
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP TC solver (%d threads) ...\n", num_threads);
	Timer t;
	t.Start();
	uint64_t total_num = 0;
	#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 1)
	for (int src = 0; src < m; src ++) {
		IndexT row_begin_src = row_offsets[src];
		IndexT row_end_src = row_offsets[src+1]; 
		for (IndexT offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			IndexT dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			IndexT row_begin_dst = row_offsets[dst];
			IndexT row_end_dst = row_offsets[dst+1];
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
	total = total_num;
	t.Stop();
	printf("\truntime [omp_base] = %f sec\n", t.Seconds());
	return;
}
