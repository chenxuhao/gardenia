// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include "common.h"
#include "timer.h"
#include <vector>
#include <algorithm>
/*
static ValueType DEFAULT_RELATIVE_TOL = 1e-4;
static ValueType DEFAULT_ABSOLUTE_TOL = 1e-4;

template<typename T>
bool almost_equal(const T& a, const T& b, const ValueType a_tol, const ValueType r_tol) {
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
    if(abs(double(a - b)) > r_tol * (abs(double(a)) + abs(double(b))) + a_tol)
        return false;
    else
        return true;
}

template <typename T>
bool check_almost_equal(const T * A, const T * B, const int N) {
	bool is_almost_equal = true;
	for(int i = 0; i < N; i++) {
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
/*
template <typename T>
bool check_almost_equal(const T * A, const T * B, const int N) {
	int num_errors = 0;
	for (int n = 0; n < N; n ++) {
		if (fabs(B[n] - A[n]) > 0.0000001) {
		//if (scores[n] != scores_to_test[n]) {
			num_errors ++;
		}
	}
	//printf("num_errors = %d\n", num_errors);
	if(num_errors == 0) return true;
	else return false;
}
//*/

// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
void BCVerifier(int m, int source, int *row_offsets, int *column_indices, int num_iters, ScoreT *scores_to_test) {
	printf("Verifying...\n");
	vector<ScoreT> scores(m, 0);
	//std::cout << setiosflags(ios::fixed);
	Timer t;
	t.Start();
	for (int iter=0; iter < num_iters; iter++) {
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
		for (int depth = static_cast<int>(verts_at_depth.size()) - 1; depth >= 0; depth --) {
			for (unsigned id = 0; id < verts_at_depth[depth].size(); id ++) {
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
	//std::cout << setprecision(8) << "max_score = " << biggest_score << "\n";
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
	//for(int i = 0; i < 10; i++) 
	//	printf("score[%d]=%f, score_test[%d]=%f\n", i, scores[i], i, scores_to_test[i]);

	// Compare scores
	if(!check_almost_equal<ScoreT>(scores_to_test, scores.data(), m))
		printf("POSSIBLE FAILURE\n");
	else
		printf("Correct\n");
	return;
}
