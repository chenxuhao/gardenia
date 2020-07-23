// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"

void SpmvVerifier(Graph &g, const ValueT* Ax, const ValueT *x, ValueT *h_y, ValueT *test_y) {
	printf("Verifying...\n");
  auto m = g.V();
  auto nnz = g.E();
	auto Ap = g.in_rowptr();
	auto Aj = g.in_colidx();	
	ValueT *y = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y[i] = h_y[i];
	Timer t;
	t.Start();
	SpmvSerial(m, nnz, Ap, Aj, Ax, x, y);
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	ValueT max_error = maximum_relative_error(test_y, y, m);
	printf("\t[max error %9f]\n", max_error);
	//for(int i = 0; i < m; i++) printf("test_y[%d] = %f, y[%d] = %f\n", i, test_y[i], i, y[i]);
	if ( max_error > 5 * std::sqrt( std::numeric_limits<ValueT>::epsilon() ) )
		printf("POSSIBLE FAILURE\n");
	else
		printf("Correct\n");
}
