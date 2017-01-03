#include<algorithm>
// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
void BCVerifier(int m, int *row_offsets, int *column_indices, int num_iters, ScoreT *scores_to_test) {
	vector<ScoreT> scores(m, 0);
	for (int i = 0; i < m; i ++) scores[i] = 0;
	for (int iter=0; iter < num_iters; iter++) {
		int source = 0;
		// BFS phase, only records depth & path_counts
		vector<int> depths(m, -1);
		depths[source] = 0;
		vector<int> path_counts(m, 0);
		path_counts[source] = 1;
		vector<int> to_visit;
		to_visit.reserve(m);
		to_visit.push_back(source);
		for (vector<int>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
			int src = *it;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				if (depths[dst] == -1) {
					depths[dst] = depths[src] + 1;
					to_visit.push_back(dst);
				}
				if (depths[dst] == depths[src] + 1)
					path_counts[dst] += path_counts[src];
			}
		}
		// Get lists of vertices at each depth
		vector<vector<int> > verts_at_depth;
		for (int n = 0; n < m; n ++) {
			if (depths[n] != -1) {
				if (depths[n] >= static_cast<int>(verts_at_depth.size()))
					verts_at_depth.resize(depths[n] + 1);
				verts_at_depth[depths[n]].push_back(n);
			}
		}
		// Going from farthest to clostest, compute "depencies" (deltas)
		vector<ScoreT> deltas(m, 0);
		for (int depth = verts_at_depth.size() - 1; depth >= 0; depth --) {
			for (int id = 0; id < verts_at_depth[depth].size(); id ++) {
				int src = verts_at_depth[depth][id];
				int row_begin = row_offsets[src];
				int row_end = row_offsets[src + 1];
				for (int offset = row_begin; offset < row_end; offset ++) {
					int dst = column_indices[offset];
					if (depths[dst] == depths[src] + 1) {
						deltas[src] += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				scores[src] += deltas[src];
			}
		}
	}
	// Normalize scores
	ScoreT biggest_score = *max_element(scores.begin(), scores.end());
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	// Compare scores
	bool all_ok = true;
	for (int n = 0; n < m; n ++) {
		if (scores[n] != scores_to_test[n]) {
			cout << n << ": " << scores[n] << " != " << scores_to_test[n] << endl;
			all_ok = false;
		}
	}
	if(all_ok) printf("Correct\n");
	return;
}
