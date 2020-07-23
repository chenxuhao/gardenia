// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define SPMV_VARIANT "omp_target"

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degrees) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	//printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);
	warm_up();
	Timer t;
	t.Start();
	double t1, t2;
	#pragma omp target data device(0) map(tofrom:y[0:m]) map(to:m,Ap[0:(m+1)],x[0:m]) map(to:Aj[0:nnz],Ax[0:nnz])
	//#pragma omp target device(0) map(tofrom:y[0:m]) map(to:m,Ap[0:(m+1)],x[0:m]) map(to:Aj[0:nnz],Ax[0:nnz])
	{
	t1 = omp_get_wtime();
	#pragma omp target device(0)
	#pragma omp parallel for
	for (int i = 0; i < m; i++){
		int row_begin = Ap[i];
		int row_end   = Ap[i+1];
		ValueT sum = y[i];
		#pragma ivdep
		//#pragma omp simd
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t2 = omp_get_wtime();
	}
	t.Stop();
	//printf("\ttotal runtime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, 1000*(t2-t1));
	return;
}
