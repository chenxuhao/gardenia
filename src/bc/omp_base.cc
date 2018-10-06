// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include <omp.h>
#include <vector>
//#include <algorithm>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

#define BC_VARIANT "omp_base"

void PBFS(int m, IndexT *row_offsets, IndexT *column_indices, int source, vector<int> &path_counts,  vector<int> &depths,
	Bitmap &succ, vector<SlidingQueue<IndexT>::iterator> &depth_index, SlidingQueue<IndexT> &queue) {
	depths[source] = 0;
	path_counts[source] = 1;
	queue.push_back(source);
	depth_index.push_back(queue.begin());
	queue.slide_window();
	#pragma omp parallel
	{
		int depth = 0;
		QueueBuffer<IndexT> lqueue(queue);
		while (!queue.empty()) {
			#pragma omp single
			depth_index.push_back(queue.begin());
			depth++;
			#pragma omp for schedule(dynamic, 64)
			for (IndexT *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
				IndexT src = *q_iter;
				IndexT row_begin = row_offsets[src];
				IndexT row_end = row_offsets[src + 1];
				for (IndexT offset = row_begin; offset < row_end; offset ++) {
					IndexT dst = column_indices[offset];
					if (depths[dst] == -1  && (compare_and_swap(depths[dst], -1, depth))) {
						lqueue.push_back(dst);
					}
					if (depths[dst] == depth) {
						succ.set_bit_atomic(offset);
						fetch_and_add(path_counts[dst], path_counts[src]);
					}
				}
			}
			lqueue.flush();
			#pragma omp barrier
			#pragma omp single
			queue.slide_window();
		}
	}
	depth_index.push_back(queue.begin());
}

void BCSolver(int m, int nnz, int source, IndexT *row_offsets, IndexT *column_indices, ScoreT *scores) {
	//omp_set_num_threads(12);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BC solver (%d threads)...\n", num_threads);
	int num_iters = 1;
	Bitmap succ(nnz);
	vector<SlidingQueue<IndexT>::iterator> depth_index;

	Timer t;
	t.Start();
	SlidingQueue<IndexT> queue(m);
	for (int iter = 0; iter < num_iters; iter++) {
		vector<int> path_counts(m, 0);
		vector<int> depths(m, -1);
		depth_index.resize(0);
		queue.reset();
		succ.reset();
		PBFS(m, row_offsets, column_indices, source, path_counts, depths, succ, depth_index, queue);
		//printf("depth_index_size = %ld\n", depth_index.size());
		vector<ScoreT> deltas(m, 0);
		for (int d = depth_index.size()-2; d >= 0; d --) {
			#pragma omp parallel for schedule(dynamic, 64)
			for (IndexT *it = depth_index[d]; it < depth_index[d+1]; it++) {
				IndexT src = *it;
				IndexT row_begin = row_offsets[src];
				IndexT row_end = row_offsets[src + 1];
				for (IndexT offset = row_begin; offset < row_end; offset ++) {
					IndexT dst = column_indices[offset];
					//if (depths[dst] == depths[src] + 1) {
					if (succ.get_bit(offset)) {
						deltas[src] += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				scores[src] += deltas[src];
			}
		}
	}
	// Normalize scores
	ScoreT biggest_score = 0;
	#pragma omp parallel for reduction(max : biggest_score)
	for (int n = 0; n < m; n ++)
		biggest_score = max(biggest_score, scores[n]);
	#pragma omp parallel for
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	return;
}
