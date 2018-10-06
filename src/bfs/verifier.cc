// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <iostream>
#include <vector>
#include "bfs.h"
#include "timer.h"
void BFSVerifier(int m, int source, IndexT *row_offsets, IndexT *column_indices, DistT *depth_to_test) {
	printf("Verifying...\n");
	vector<DistT> depth(m, MYINFINITY);
	vector<int> to_visit;
	Timer t;
	t.Start();
	depth[source] = 0;
	to_visit.reserve(m);
	to_visit.push_back(source);
	for (std::vector<int>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
		int src = *it;
		const IndexT row_begin = row_offsets[src];
		const IndexT row_end = row_offsets[src + 1]; 
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT dst = column_indices[offset];
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
/*
void write_solution(const char *fname, int m, DistT *dist) {
	assert(dist != NULL);
	printf("Writing solution to %s\n", fname);
	FILE *f = fopen(fname, "w");
	fprintf(f, "Computed solution (source dist): [");
	for(int n = 0; n < m; n ++) {
		fprintf(f, "%d:%d\n ", n, dist[n]);
	}
	fprintf(f, "]");
}
*/
