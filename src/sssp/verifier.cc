// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <queue>
#include <iostream>
#include "sssp.h"
#include "timer.h"

void SSSPVerifier(Graph &g, int source, DistT *weight, DistT *dist_to_test) {
	printf("Verifying...\n");
	// Serial Dijkstra implementation to get oracle distances
	vector<DistT> oracle_dist(g.V(), kDistInf);
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
      auto offset = g.edge_begin(src);
      for (auto dst : g.N(src)) {
				DistT wt = weight[offset++];
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
	for (int n = 0; n < g.V(); n ++) {
		if (dist_to_test[n] != oracle_dist[n]) {
			//std::cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << std::endl;
			all_ok = false;
		}
	}
	if(all_ok) printf("Correct\n");
	else printf("Wrong\n");
}

