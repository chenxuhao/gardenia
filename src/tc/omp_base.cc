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
	for (int u = 0; u < m; u ++) {
		IndexT begin_u = row_offsets[u];
		IndexT end_u = row_offsets[u+1]; 
		for (IndexT e_u = begin_u; e_u < end_u; ++ e_u) {
			IndexT v = column_indices[e_u];
			int it = begin_u;
			IndexT begin_v = row_offsets[v];
			IndexT end_v = row_offsets[v+1];
			for (IndexT e_v = begin_v; e_v < end_v; ++ e_v) {
				IndexT w = column_indices[e_v];
				while (column_indices[it] < w && it < end_u)
					it ++;
				if (it != end_u && w == column_indices[it]) {
          //std::cout << "u = " << u << " v = " << v << " w = " << w << "\n";
					total_num ++;
        }
			}
		} 
	}
	total = total_num;
	t.Stop();
	printf("\truntime [omp_base] = %f sec\n", t.Seconds());
	return;
}
