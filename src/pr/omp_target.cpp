// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define PR_VARIANT "omp_target"

void PRSolver(int m, int nnz, int *row_offsets, int *column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *scores) {
	double t1, t2;
	const ScoreT base_score = (1.0f - kDamp) / m;
	ScoreT *outgoing_contrib = (ScoreT *) malloc(m*sizeof(ScoreT));
	warm_up();
	int iter;
	Timer t;
	t.Start();
#pragma omp target data device(0) map(tofrom:scores[0:m]) map(to:row_offsets[0:(m+1)]) map(to:degree[0:m]) map(to:outgoing_contrib[0:m]) map(to:column_indices[0:nnz]) map(to:base_score)
{
	t1 = omp_get_wtime();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		#pragma omp target device(0)
		#pragma omp parallel for
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
		#pragma omp target device(0)
		#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			//#pragma omp simd reduction(+ : incoming_total)
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t2 = omp_get_wtime();
}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	//printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, 1000*(t2-t1));
	return;
}
