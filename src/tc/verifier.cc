// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include "timer.h"
#include <vector>
#include <algorithm>

void TCVerifier(const Graph &g, unsigned long long test_total) {
	printf("Verifying...\n");
	unsigned long long total = 0;
	Timer t;
	t.Start();
	for (VertexID u=0; u < g.num_vertices(); u++) {
		for (VertexID v : g.out_neigh(u)) {
			if (v > u) break;
			auto it = g.out_neigh(u).begin();
			for (VertexID w : g.out_neigh(v)) {
				if (w > v) break;
				while (*it < w) it++;
				if (w == *it) total++;
			}
		}
	}
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());
	if(total == test_total) printf("Correct\n");
	else printf("Wrong\n");
	std::cout << "total=" << total << "test_total=" << test_total << std::endl;
	return;
}
