// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include <omp.h>
#include "timer.h"
#ifdef SIM
#include "sim.h"
#include "bitmap.h"
#endif
#define CC_VARIANT "omp_base"

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexType *row_offsets, IndexType *column_indices, int *degree, CompT *comp, bool is_directed) {
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
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
#ifdef SIM
	//int *hub_bitmap = (int *) malloc(m * sizeof(int));
	Bitmap hub(m);
	hub.reset();
	int num_hubs = set_hub(m, nnz, degree, hub);
	m5_checkpoint(0,0);
	set_addr_bounds(1,(uint64_t)row_offsets,(uint64_t)&row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)column_indices,(uint64_t)&column_indices[nnz],8);
	set_addr_bounds(3,(uint64_t)comp,(uint64_t)&comp[m],8);
	set_addr_bounds(6,(uint64_t)hub.start_,(uint64_t)hub.end_,8);
	printf("Begin of ROI\n");
	printf("This graph has %d hub vertices\n", num_hubs);
#endif
	while (change) {
		change = false;
		iter++;
		printf("Executing iteration %d ...\n", iter);
		#pragma omp parallel for schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			CompT comp_src = comp[src];
			IndexType row_begin = row_offsets[src];
			IndexType row_end = row_offsets[src+1];
			for (IndexType offset = row_begin; offset < row_end; offset ++) {
				IndexType dst = column_indices[offset];
				CompT comp_dst = comp[dst];
				if (comp_src == comp_dst) continue;
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
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	return;
}
