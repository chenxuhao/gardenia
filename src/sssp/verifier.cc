// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <queue>
#include <iostream>
#include "sssp.h"
#include "timer.h"
void SSSPVerifier(int m, int source, IndexT *row_offsets, IndexT *column_indices, DistT *weight, DistT *dist_to_test) {
	printf("Verifying...\n");
	// Serial Dijkstra implementation to get oracle distances
	vector<DistT> oracle_dist(m, kDistInf);
	typedef pair<DistT, IndexT> WN;
	priority_queue<WN, vector<WN>, greater<WN> > mq;
	int iter = 0;
	Timer t;
	t.Start();
	oracle_dist[source] = 0;
	mq.push(make_pair(0, source));
	while (!mq.empty()) {
		DistT td = mq.top().first;
		IndexT src = mq.top().second;
		mq.pop();
		if (td == oracle_dist[src]) {
			IndexT row_begin = row_offsets[src];
			IndexT row_end = row_offsets[src + 1]; 
			for (IndexT offset = row_begin; offset < row_end; ++ offset) {
				IndexT dst = column_indices[offset];
				DistT wt = weight[offset];
				if (td + wt < oracle_dist[dst]) {
					oracle_dist[dst] = td + wt;
					mq.push(make_pair(td + wt, dst));
				}
			}
		}
		iter ++;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());

	// Report any mismatches
	bool all_ok = true;
	for (int n = 0; n < m; n ++) {
		if (dist_to_test[n] != oracle_dist[n]) {
			//std::cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << std::endl;
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
