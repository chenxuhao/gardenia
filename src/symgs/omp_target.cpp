// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "symgs.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define SYMGS_VARIANT "omp_target"

void gauss_seidel(int *Ap, int *Aj, int *indices, ValueT *Ax, ValueT *x, ValueT *b, int row_start, int row_stop, int row_step) {
	#pragma omp target device(0)
	#pragma omp parallel for
	for (int i = row_start; i < row_stop; i += row_step) {
		int inew = indices[i];
		int row_begin = Ap[inew];
		int row_end = Ap[inew+1];
		ValueT rsum = 0;
		ValueT diag = 0;
		#pragma ivdep
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			if (inew == j) diag = Ax[jj];
			else rsum += x[j] * Ax[jj];
		}
		if (diag != 0) x[inew] = (b[inew] - rsum) / diag;
	}
}

void SymGSSolver(int num_rows, int nnz, int *Ap, int *Aj, int *indices, ValueT *Ax, ValueT *x, ValueT *b, std::vector<int> color_offsets) {
	warm_up();
	Timer t;
	t.Start();
	double t1, t2;
	#pragma omp target data device(0) map(tofrom:x[0:num_rows]) map(to:Ap[0:(num_rows+1)],b[0:num_rows],indices[0:num_rows]) map(to:Aj[0:nnz],Ax[0:nnz])
{
	t1 = omp_get_wtime();
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i], color_offsets[i+1], 1);
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i-1], color_offsets[i], 1);
	t2 = omp_get_wtime();
}
	t.Stop();
	//printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, 1000*(t2-t1));
	return;
}
