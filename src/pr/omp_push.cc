// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include <stdlib.h>
#include <atomic>
#include <vector>
#include "timer.h"
#include "platform_atomics.h"
#define PR_VARIANT "omp_push"

void PRSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degree, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	//ScoreT *sums = (ScoreT *) malloc(m * sizeof(ScoreT));
	//for (int i = 0; i < m; i ++) { sums[i] = base_score; }
	vector<ScoreT> sums(m, 0);
	//vector<std::atomic<ScoreT>> sums(m, 0);
	int iter;
	Timer t;
	t.Start();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		#pragma omp parallel for schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			const IndexT row_begin = out_row_offsets[src];
			const IndexT row_end = out_row_offsets[src + 1];
			int degree = row_end - row_begin;
			ScoreT contribution = scores[src] / (ScoreT)degree;
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = out_column_indices[offset];
				#pragma omp atomic
				sums[dst] = sums[dst] + contribution;
				//fetch_and_add(sums[dst], contribution);
				//sums[dst].fetch_add(contribution, std::memory_order_relaxed);
			}
		}
		double error = 0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++) {
			ScoreT new_score = base_score + kDamp * sums[u];
			error += fabs(new_score - scores[u]);
			scores[u] = new_score;
			sums[u] = 0;
		}
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
