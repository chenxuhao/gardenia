// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "scc.h"
#include <omp.h>
#include "timer.h"
#define SCC_VARIANT "openmp"

void SCCSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *color) {
	printf("Launching OpenMP SCC solver...\n");
	//omp_set_num_threads(2);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	int iter = 0;
	Timer t;
	t.Start();

	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SCC_VARIANT, t.Millisecs());
	return;
}
