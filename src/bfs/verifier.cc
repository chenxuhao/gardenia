// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <iostream>
#include <vector>
#include "bfs.h"
#include "timer.h"

void BFSVerifier(Graph &g, int source, DistT *depth_to_test) {
	std::cout << "Verifying...\n";
  auto m = g.V();
	vector<DistT> depth(m, MYINFINITY);
	vector<int> to_visit;
	Timer t;
	t.Start();
	depth[source] = 0;
	to_visit.reserve(m);
	to_visit.push_back(source);
	for (std::vector<int>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
		int src = *it;
    for (auto dst : g.N(src)) {
			if (depth[dst] == MYINFINITY) {
				depth[dst] = depth[src] + 1;
				to_visit.push_back(dst);
			}
		}
	}
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	// Report any mismatches
	bool all_ok = true;
	for (int n = 0; n < m; n ++) {
		if (depth_to_test[n] != depth[n]) {
			//std::cout << n << ": " << depth_to_test[n] << " != " << depth[n] << std::endl;
			all_ok = false;
		}
	}
	if(all_ok) printf("Correct\n");
	else printf("Wrong\n");
}

