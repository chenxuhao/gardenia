#include "cc.h"
#include <omp.h>
#include "timer.h"
#define CC_VARIANT "openmp"

void CCSolver(int m, int nnz, int *row_offsets, int *column_indices, CompT *comp) {
	printf("Launching OpenMP CC solver...\n");
	omp_set_num_threads(2);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	Timer t;
	t.Start();
#pragma omp parallel for
	for (int n=0; n < m; n++)
		comp[n] = n;
	bool change = true;
	int num_iter = 0;
	while (change) {
		change = false;
		num_iter++;
#pragma omp parallel for
		for (int src = 0; src < m; src ++) {
			int comp_src = comp[src];
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				int comp_dst = comp[dst];      
				if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
					change = true;
					comp[comp_dst] = comp_src;
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
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	return;
}
