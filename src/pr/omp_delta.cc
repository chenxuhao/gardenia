// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include <stdlib.h>
#include <atomic>
#include <vector>
#include "timer.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define PR_VARIANT "omp_delta"

void push_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *deltas, ScoreT *sums, SlidingQueue<IndexT> &queue) {
	#pragma omp parallel for schedule(dynamic, 64)
	for (IndexT *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
	//for (IndexT src = 0; src < m; src ++) {
		IndexT src = *q_iter;
		const IndexT row_begin = row_offsets[src];
		const IndexT row_end = row_offsets[src+1];
		int degree = row_end - row_begin;
		ScoreT contribution = deltas[src] / (ScoreT)degree;
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			#pragma omp atomic
			sums[dst] += contribution;
		}
	}

}

void pull_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *deltas, ScoreT *sums, SlidingQueue<IndexT> &queue, int *degrees) {
	vector<ScoreT> outgoing_contrib(m, 0);
	#pragma omp parallel for
	for (int n = 0; n < m; n ++)
		outgoing_contrib[n] = deltas[n] / degrees[n];
	#pragma omp parallel for schedule(dynamic, 64)
	for (int dst = 0; dst < m; dst ++) {
		ScoreT incoming_total = 0;
		const IndexT row_begin = row_offsets[dst];
		const IndexT row_end = row_offsets[dst+1];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT src = column_indices[offset];
			incoming_total += outgoing_contrib[src];
		}
		sums[dst] = incoming_total;
	}
}

void PRSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	vector<ScoreT> sums(m, 0);
	vector<ScoreT> deltas(m, init_score);
	int iter = 0;
	Timer t;
	t.Start();
	SlidingQueue<IndexT> queue(m);
	for (int i = 0; i < m; i ++) queue.push_back(i);
	queue.slide_window();
	while (!queue.empty() && iter < MAX_ITER) {
		++ iter;
		//printf("iteration=%d, num_frontier=%ld\n", iter, queue.size());
		if (queue.size() < m/10) {
		//if (0) {
			printf("push:");
			push_step(m, out_row_offsets, out_column_indices, deltas.data(), sums.data(), queue);
		} else {
			printf("pull:");
			pull_step(m, row_offsets, column_indices, deltas.data(), sums.data(), queue, degrees);
		}
		queue.reset();
		#pragma omp parallel
		{
		QueueBuffer<IndexT> lqueue(queue);
		#pragma omp for
		for (int u = 0; u < m; u ++) {
			if (iter == 1) {
				deltas[u] = base_score + kDamp * sums[u];
				deltas[u] -= init_score;
			} else {
				deltas[u] = kDamp * sums[u];
			}
			scores[u] += deltas[u];
			sums[u] = 0;
			if (fabs(deltas[u]) > epsilon2 * scores[u]) 
				lqueue.push_back(u);
		}
		lqueue.flush();
		}
		queue.slide_window();
		double error = 0.0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++)
			error += fabs(deltas[u]);
		printf(" %2d    %lf\n", iter, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
