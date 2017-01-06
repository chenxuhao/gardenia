#include "bc.h"
#include <omp.h>
#include "timer.h"
#define BC_VARIANT "openmp"
void BCSolver(int m, int nnz, int *row_offsets, int *column_indices, ScoreT *scores, int device) {
	printf("Launching OpenMP BC solver...\n");
	omp_set_num_threads(2);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	Timer t;
	t.Start();
	for(int i=0; i<m; i++) scores[i] = 0;
	//for (int iter=0; iter < num_iters; iter++) {
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
			#pragma omp parallel for schedule(dynamic, 64)
			for (int id = 0; id < verts_at_depth[depth].size(); id ++) {
				int src = verts_at_depth[depth][id];
				int row_begin = row_offsets[src];
				int row_end = row_offsets[src + 1];
				for (int offset = row_begin; offset < row_end; offset ++) {
					int dst = column_indices[offset];
			if(src==237) printf("dst %d: depth=%d, path_counts=%d, delta=%.8f, accu=%.8f\n", dst, depths[dst], path_counts[dst], deltas[dst], static_cast<ScoreT>(path_counts[src]) / static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]));
					if (depths[dst] == depths[src] + 1) {
						deltas[src] += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				scores[src] += deltas[src];
		if(src==237) printf("Vertex %d: depth=%d, out_degree=%d, path_count=%d, delta=%.8f, score=%.8f\n", src, depths[src], row_end-row_begin, path_counts[src], deltas[src], scores[src]);
			}
		}
	//}

	// Normalize scores
	ScoreT biggest_score = 0;
	#pragma omp parallel for reduction(max : biggest_score)
	for (int n = 0; n < m; n ++)
		biggest_score = max(biggest_score, scores[n]);
	#pragma omp parallel for
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	//for (int i = 0; i < 10; i++) printf("scores[%d] = %.8f\n", i, scores[i]);
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	return;
}
