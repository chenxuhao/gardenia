// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#define SPMV_VARIANT "omp_push"

// m: number of vertices, nnz: number of non-zero values
void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degrees) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();
	#pragma omp parallel for schedule(dynamic, 1024)
	for (int u = 0; u < m; u ++) {
		//int tid = omp_get_thread_num();
		IndexT row_begin = ApT[u];
		IndexT row_end = ApT[u+1];
		ScoreT c = x[u];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT v = AjT[offset];
			ValueT value = AxT[offset];
			#pragma omp atomic
			y[v] += c * value;
		}
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}

