// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"

void SpmvSolver(Graph &g, const ValueT* Ax, const ValueT *x, ValueT *y) {
  auto m = g.V();
  auto nnz = g.E();
	auto Ap = g.in_rowptr();
	auto Aj = g.in_colidx();
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();

	#pragma omp parallel for
	for (int i = 0; i < m; i++){
		auto row_begin = Ap[i];
		auto row_end   = Ap[i+1];
		ValueT sum = y[i];
		//#pragma omp simd reduction(+:sum)
		for (auto jj = row_begin; jj < row_end; jj++) {
			auto j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	printf("\truntime [omp_base] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", time, GFLOPs, GBYTEs);
	return;
}

