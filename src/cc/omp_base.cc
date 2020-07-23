// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#define CC_VARIANT "omp_base"

void CCSolver(Graph &g, CompT *comp) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP CC solver (%d threads) ...\n", num_threads);
	#pragma omp parallel for
	for (int n = 0; n < g.V(); n ++) comp[n] = n;
	bool change = true;
	int iter = 0;

	Timer t;
	t.Start();
	while (change) {
		change = false;
		iter++;
		//printf("Executing iteration %d ...\n", iter);
		#pragma omp parallel for schedule(dynamic, 64)
		for (int src = 0; src < g.V(); src ++) {
			CompT comp_src = comp[src];
      for (auto dst : g.N(src)) {
				CompT comp_dst = comp[dst];
				if (comp_src == comp_dst) continue;
				// Hooking condition so lower component ID wins independent of direction
				int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
				int low_comp = comp_src + (comp_dst - high_comp);
				if (high_comp == comp[high_comp]) {
					change = true;
					comp[high_comp] = low_comp;
				}
			}
		}
		#pragma omp parallel for
		for (int n = 0; n < g.V(); n++) {
			while (comp[n] != comp[comp[n]]) {
				comp[n] = comp[comp[n]];
			}
		}
	}
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [omp_base] = %f ms.\n", t.Millisecs());
	return;
}
