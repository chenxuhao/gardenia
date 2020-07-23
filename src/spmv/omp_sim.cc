// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#ifdef SIM
#include "sim.h"
#include "bitmap.h"
#endif
#define SPMV_VARIANT "omp_base"

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degrees) {
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();
#ifdef SIM
	Bitmap hub(m);
	hub.reset();
	int num_hubs = set_hub(m, nnz, degrees, hub);
	m5_checkpoint(0,0);
	set_addr_bounds(1,(uint64_t)Ap,(uint64_t)&Ap[m+1],4);
	set_addr_bounds(2,(uint64_t)Aj,(uint64_t)&Aj[nnz],8);
	set_addr_bounds(3,(uint64_t)x,(uint64_t)&x[m],8);
	//set_addr_bounds(1,(uint64_t)y,(uint64_t)&y[m],4);
	set_addr_bounds(5,(uint64_t)Ax,(uint64_t)&Ax[nnz],8);
	set_addr_bounds(6,(uint64_t)hub.start_,(uint64_t)hub.end_,8);
	printf("Begin of ROI\n");
	printf("This graph has %d hub vertices\n", num_hubs);
#endif

	#pragma omp parallel for
	for (int i = 0; i < m; i++){
		const IndexType row_begin = Ap[i];
		const IndexType row_end   = Ap[i+1];
		ValueType sum = y[i];
		//#pragma omp simd reduction(+:sum)
		for (IndexType jj = row_begin; jj < row_end; jj++) {
			const IndexType j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
