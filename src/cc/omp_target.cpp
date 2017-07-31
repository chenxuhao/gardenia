// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define CC_VARIANT "omp_target"

#pragma omp declare target
void cc_scatter(int m, int *row_offsets, int *column_indices, CompT *comp, bool &change) {
	//#pragma omp target device(0)
	#pragma omp parallel for schedule(dynamic, 64)
	for (int src = 0; src < m; src ++) {
		int comp_src = comp[src];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		#pragma ivdep
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = column_indices[offset];
			int comp_dst = comp[dst];      
			if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
				change = true;
				comp[comp_dst] = comp_src;
			}
		}
	}
}

void cc_update(int m, CompT *comp) {
	//#pragma omp target device(0)
	#pragma omp parallel for
	for (int n = 0; n < m; n++) {
		int count = 0;
		while (comp[n] != comp[comp[n]]) {
			comp[n] = comp[comp[n]];
			if(count>128) break;
			count ++;
		}
	}
}
#pragma omp end declare target

void CCSolver(int m, int nnz, int *row_offsets, int *column_indices, CompT *comp) {
	#pragma omp parallel for
	for (int n=0; n < m; n++) comp[n] = n;
	Timer t;
	t.Start();
	double t1, t2;
	warm_up();
	int iter = 0;

#pragma omp target data device(0) map(tofrom:comp[0:m]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz])
{
	#pragma omp target device(0)
	{
	t1 = omp_get_wtime();
	bool change = true;
	while (change) {
		change = false;
		iter ++;
		cc_scatter(m, row_offsets, column_indices, comp, change);
		cc_update(m, comp);
	}
	t2 = omp_get_wtime();
	}
}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	//printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, 1000*(t2-t1));
	return;
}
