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
#ifdef SIM
#include "sim.h"
#endif

#define BC_VARIANT "omp_base"

void PBFS(int m, IndexType *row_offsets, IndexType *column_indices, int source, vector<int> &path_counts,  vector<int> &depths,
	Bitmap &succ, vector<SlidingQueue<IndexType>::iterator> &depth_index, SlidingQueue<IndexType> &queue) {
	depths[source] = 0;
	path_counts[source] = 1;
	queue.push_back(source);
	depth_index.push_back(queue.begin());
	queue.slide_window();
	#pragma omp parallel
	{
		int depth = 0;
		QueueBuffer<IndexType> lqueue(queue);
		while (!queue.empty()) {
			#pragma omp single
			depth_index.push_back(queue.begin());
			depth++;
			#pragma omp for schedule(dynamic, 64)
			for (IndexType *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
				IndexType src = *q_iter;
				IndexType row_begin = row_offsets[src];
				IndexType row_end = row_offsets[src + 1];
				for (IndexType offset = row_begin; offset < row_end; offset ++) {
					IndexType dst = column_indices[offset];
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

void BCSolver(int m, int nnz, int source, IndexType *row_offsets, IndexType *column_indices, ScoreT *scores) {
	//omp_set_num_threads(12);
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BC solver (%d threads)...\n", num_threads);
	int num_iters = 1;
	Bitmap succ(nnz);
	vector<SlidingQueue<IndexType>::iterator> depth_index;

	Timer t;
	t.Start();
#ifdef SIM
	m5_checkpoint(0,0);
#endif
	SlidingQueue<IndexType> queue(m);
#ifdef SIM
	set_addr_bounds(1,(uint64_t)row_offsets,(uint64_t)&row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)column_indices,(uint64_t)&column_indices[nnz],8);
	set_addr_bounds(3,(uint64_t)scores,(uint64_t)&scores[m],8);
	printf("Begin of ROI\n");
#endif
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
			for (IndexType *it = depth_index[d]; it < depth_index[d+1]; it++) {
				IndexType src = *it;
				IndexType row_begin = row_offsets[src];
				IndexType row_end = row_offsets[src + 1];
				for (IndexType offset = row_begin; offset < row_end; offset ++) {
					IndexType dst = column_indices[offset];
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
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	return;
}
