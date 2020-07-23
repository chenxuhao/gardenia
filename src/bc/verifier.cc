// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "bc.h"
#include "timer.h"
#include "common.h"
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>
/*
static ValueT DEFAULT_RELATIVE_TOL = 1e-4;
static ValueT DEFAULT_ABSOLUTE_TOL = 1e-4;

template<typename T>
bool almost_equal(const T& a, const T& b, const ValueT a_tol, const ValueT r_tol) {
    if(fabs(a - b) > r_tol * (fabs(a) + fabs(b)) + a_tol)
        return false;
    else
        return true;
}
*/
///*
static double DEFAULT_RELATIVE_TOL = 1e-4;
static double DEFAULT_ABSOLUTE_TOL = 1e-4;

template<typename T>
bool almost_equal(const T& a, const T& b, const double a_tol, const double r_tol) {
    using std::abs;
    if(fabs(double(a - b)) > r_tol * (fabs(double(a)) + fabs(double(b))) + a_tol)
        return false;
    else
        return true;
}

template <typename T>
bool check_almost_equal(int m, const T * A, const T * B) {
	bool is_almost_equal = true;
	for(int i = 0; i < m; i++) {
		const T a = A[i];
		const T b = B[i];
		if(!almost_equal(a, b, DEFAULT_ABSOLUTE_TOL, DEFAULT_RELATIVE_TOL)) {
			is_almost_equal = false;
			printf("score_test[%d] (%f) != score[%d] (%f)\n", i, A[i], i, B[i]);
			break;
		}
	}
	return is_almost_equal;
}
//*/

template <typename T>
bool check_equal(int m, const T * A, const T * B) {
	bool is_equal = true;
	for (int i = 0; i < m; i ++) {
		//if (fabs(B[i] - A[i]) > 0.0000001) {
		if (A[i] != B[i]) {
			is_equal = false;
			printf("score_test[%d] (%f) != score[%d] (%f)\n", i, A[i], i, B[i]);
			break;
		}
	}
	return is_equal;
}

// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
void BCVerifier(Graph &g, int source, int num_iters, ScoreT *scores_to_test) {
	printf("Verifying...\n");
  auto m = g.V();
	vector<ScoreT> scores(m, 0);
	//std::cout << setiosflags(ios::fixed);
	int max_depth = 0;

	Timer t;
	t.Start();
	for (int iter=0; iter < num_iters; iter++) {
		// BFS phase, only records depth & path_counts
		vector<int> depths(m, -1);
		depths[source] = 0;
		vector<int> path_counts(m, 0);
		path_counts[source] = 1;
		vector<IndexT> to_visit;
		to_visit.reserve(m);
		to_visit.push_back(source);
		for (vector<IndexT>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
			IndexT src = *it;
      for (auto dst : g.N(src)) {
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
		max_depth = static_cast<int>(verts_at_depth.size());
		// Going from farthest to clostest, compute "depencies" (deltas)
		vector<ScoreT> deltas(m, 0);
		for (int depth = max_depth - 1; depth >= 0; depth --) {
			for (unsigned id = 0; id < verts_at_depth[depth].size(); id ++) {
				int src = verts_at_depth[depth][id];
				ScoreT delta_src = 0;
        for (auto dst : g.N(src)) {
					if (depths[dst] == depths[src] + 1) {
						delta_src += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				deltas[src] = delta_src;
				scores[src] += delta_src;
			}
		}
	}
	
	// Normalize scores
	ScoreT biggest_score = *max_element(scores.begin(), scores.end());
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	t.Stop();

	printf("\titerations = %d.\n", max_depth);
	//std::cout << "\t" << setprecision(8) << "max_score = " << biggest_score << "\n";
	printf("\tmax_score = %.6f.\n", biggest_score);
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
	//for(int i = 0; i < 10; i++) 
	//	printf("score[%d]=%f, score_test[%d]=%f\n", i, scores[i], i, scores_to_test[i]);
	// Compare scores
	if(!check_almost_equal<ScoreT>(m, scores_to_test, scores.data()))
	//if(!check_equal<ScoreT>(m, scores_to_test, scores.data()))
		printf("POSSIBLE FAILURE\n");
	else
		printf("Correct\n");
	return;
}
