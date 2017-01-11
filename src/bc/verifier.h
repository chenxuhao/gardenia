#include<algorithm>
// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
void BCVerifier(int m, int *row_offsets, int *column_indices, int num_iters, ScoreT *scores_to_test) {
	vector<ScoreT> scores(m, 0);
	vector<int> depths(m, -1);
	for (int iter=0; iter < num_iters; iter++) {
		int source = 0;
		// BFS phase, only records depth & path_counts
		//vector<int> depths(m, -1);
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
		//for (int i = 0; i < 10; i++) printf("path_counts[%d] = %d\n", i, path_counts[i]);
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
		assert(verts_at_depth.size() > 1); // the graph has more than one vertex
		for (int depth = static_cast<int>(verts_at_depth.size()) - 1; depth >= 0; depth --) {
			//printf("In depth %d (%ld):\n", depth, verts_at_depth[depth].size());
			for (unsigned id = 0; id < verts_at_depth[depth].size(); id ++) {
				int src = verts_at_depth[depth][id];
				int row_begin = row_offsets[src];
				int row_end = row_offsets[src + 1];
				for (int offset = row_begin; offset < row_end; offset ++) {
					int dst = column_indices[offset];
					//if(src==237) printf("\tdst %d: depth=%d, path_counts=%d, delta=%.8f, accu=%.8f\n", dst, depths[dst], path_counts[dst], deltas[dst], static_cast<ScoreT>(path_counts[src]) / static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]));
					if (depths[dst] == depths[src] + 1) {
						deltas[src] += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				scores[src] += deltas[src];
				//if(src==237) printf("Vertex %d: depth=%d, out_degree=%d, path_count=%d, delta=%.8f, score=%.8f\n", src, depths[src], row_end-row_begin, path_counts[src], deltas[src], scores[src]);
			}
		}
	}
	//for (int i = 0; i < 10; i++) printf("scores[%d] = %.8f\n", i, scores[i]);
	// Normalize scores
	ScoreT biggest_score = *max_element(scores.begin(), scores.end());
	//printf("max_score = %f\n", biggest_score);
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	//for (int i = 0; i < 10; i++) printf("scores[%d] = %.8f\n", i, scores[i]);
	// Compare scores
	int num_errors = 0;
	for (int n = 0; n < m; n ++) {
		//if (fabs(scores[n] - scores_to_test[n]) > 0.0000001) {
		if (scores[n] != scores_to_test[n]) {
			if(depths[n]>15 && num_errors<10) printf("Vertex %d (depth=%d): %.8f != %.8f\n", n, depths[n], scores[n], scores_to_test[n]);
			num_errors ++;
		}
	}
	if(num_errors == 0) printf("Correct\n");
	else printf("Wrong: num_errors = %d\n", num_errors);
	return;
}
