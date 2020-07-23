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

void PBFS(Graph &g, int source, vector<int> &path_counts,  vector<int> &depths,
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
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
					if (depths[dst] == -1  && (compare_and_swap(depths[dst], -1, depth))) {
						lqueue.push_back(dst);
					}
					if (depths[dst] == depth) {
						succ.set_bit_atomic(offset);
						fetch_and_add(path_counts[dst], path_counts[src]);
					}
          offset ++;
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

void BCSolver(Graph &g, int source, ScoreT *scores) {
	auto m = g.V();
	auto nnz = g.E();
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
		PBFS(g, source, path_counts, depths, succ, depth_index, queue);
		vector<ScoreT> deltas(m, 0);
		for (int d = depth_index.size()-2; d >= 0; d --) {
			#pragma omp parallel for schedule(dynamic, 64)
			for (IndexT *it = depth_index[d]; it < depth_index[d+1]; it++) {
				IndexT src = *it;
				ScoreT delta_src = 0;
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
					if (succ.get_bit(offset)) {
						delta_src += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
          offset ++;
				}
				deltas[src] = delta_src;
				scores[src] += delta_src;
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
