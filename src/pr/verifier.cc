// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
#include "pr.h"
#include "timer.h"
void PRVerifier(int m, int *row_offsets, int *column_indices, int *degree, float *scores, double target_error) {
	printf("Verifying...\n");
	const float base_score = (1.0f - kDamp) / m;
	float *incomming_sums = (float *)malloc(m * sizeof(float));
	for(int i = 0; i < m; i ++) incomming_sums[i] = 0;
	double error = 0;
	Timer t;
	t.Start();
	for (int src = 0; src < m; src ++) {
		float outgoing_contrib = scores[src] / degree[src];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			incomming_sums[dst] += outgoing_contrib;
		}
	}
	for (int i = 0; i < m; i ++) {
		float new_score = base_score + kDamp * incomming_sums[i];
		error += fabs(new_score - scores[i]);
		incomming_sums[i] = 0;
	}
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
	if (error < target_error) printf("Correct\n");
	else printf("Total Error: %f\n", error);
}

