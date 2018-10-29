// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include <stdlib.h>
#include "timer.h"
#define PR_VARIANT "omp_base"

void PRSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	ScoreT *outgoing_contrib = (ScoreT *) malloc(m * sizeof(ScoreT));
	int iter;
	Timer t;
	t.Start();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		#pragma omp parallel for
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degrees[n];
		#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (int dst = 0; dst < m; dst ++) {
			ScoreT incoming_total = 0;
			const IndexT row_begin = row_offsets[dst];
			const IndexT row_end = row_offsets[dst+1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT src = column_indices[offset];
				incoming_total += outgoing_contrib[src];
			}
			ScoreT old_score = scores[dst];
			scores[dst] = base_score + kDamp * incoming_total;
			error += fabs(scores[dst] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
