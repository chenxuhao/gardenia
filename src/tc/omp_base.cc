// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"
#include "timer.h"

void TCSolver(Graph &g, uint64_t &total) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP TC solver (%d threads) ...\n", num_threads);
	Timer t;
	t.Start();
	uint64_t total_num = 0;
	#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 1)
	for (VertexId u = 0; u < g.V(); u ++) {
    auto yu = g.N(u);
    for (auto v : yu) {
      total_num += (uint64_t)intersection_num(yu, g.N(v));
		} 
	}
	total = total_num;
	t.Stop();
	printf("\truntime [omp_base] = %f sec\n", t.Seconds());
	return;
}

