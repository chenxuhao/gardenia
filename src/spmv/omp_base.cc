// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#define TC_VARIANT "openmp"
void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
	printf("Launching OpenMP TC solver...\n");
	//omp_set_num_threads(8);
	int num_threads = 1;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	Timer t;
	t.Start();
#pragma omp parallel for //reduction(+ : sum) schedule(dynamic, 64)
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
	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	return;
}
