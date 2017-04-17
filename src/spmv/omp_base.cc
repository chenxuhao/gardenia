// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#define SPMV_VARIANT "omp_base"
void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
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
		int row_begin = Ap[i];
		int row_end   = Ap[i+1];
		ValueType sum = y[i];
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
