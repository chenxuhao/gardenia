// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#define SPMV_VARIANT "omp_base"

void SpmvSolver(int num_rows, int nnz, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degree) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();

	#pragma omp parallel for
	for (int i = 0; i < num_rows; i++){
		const IndexT row_begin = Ap[i];
		const IndexT row_end   = Ap[i+1];
		ValueT sum = y[i];
		//#pragma omp simd reduction(+:sum)
		for (IndexT jj = row_begin; jj < row_end; jj++) {
			const IndexT j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
