// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include "timer.h"
#define PR_VARIANT "openmp"

void PRSolver(int m, int nnz, int *row_offsets, int *column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *scores) {
	printf("Launching OpenMP PR solver...\n");
	//omp_set_num_threads(1);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	const ScoreT init_score = 1.0f / m;
	const ScoreT base_score = (1.0f - kDamp) / m;
	for (int i = 0; i < m; i ++) scores[i] = init_score;
	vector<ScoreT> outgoing_contrib(m);
	Timer t;
	t.Start();
	for (int iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
#pragma omp parallel for
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter, error);
		if (error < EPSILON)
			break;
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
