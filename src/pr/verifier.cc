// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
#include "pr.h"
#include "timer.h"
//#include <vector>
#include <stdlib.h>

void PRVerifier(int m, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degree, ScoreT *scores_to_test, double target_error) {
	printf("Verifying...\n");
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	ScoreT *scores = (ScoreT *) malloc(m * sizeof(ScoreT));
	for (int i = 0; i < m; i ++) scores[i] = init_score;
	ScoreT *outgoing_contrib = (ScoreT *) malloc(m * sizeof(ScoreT));
	//for (int i = 0; i < m; i ++) outgoing_contrib[i] = 0;
	int iter;
	Timer t;
	t.Start();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			const IndexT row_begin = in_row_offsets[src];
			const IndexT row_end = in_row_offsets[src + 1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = in_column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	//printf("\truntime [serial] = %f ms.\n", t.Millisecs());
	
	ScoreT *incomming_sums = (ScoreT *)malloc(m * sizeof(ScoreT));
	for(int i = 0; i < m; i ++) incomming_sums[i] = 0;
	double error = 0;
	for (int src = 0; src < m; src ++) {
		ScoreT outgoing_contrib = scores_to_test[src] / degree[src];
		const IndexT row_begin = out_row_offsets[src];
		const IndexT row_end = out_row_offsets[src + 1];
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT dst = out_column_indices[offset];
			incomming_sums[dst] += outgoing_contrib;
		}
	}
	for (int i = 0; i < m; i ++) {
		ScoreT new_score = base_score + kDamp * incomming_sums[i];
		error += fabs(new_score - scores_to_test[i]);
		incomming_sums[i] = 0;
	}
	if (error < target_error) printf("Correct\n");
	else printf("Total Error: %f\n", error);
}

