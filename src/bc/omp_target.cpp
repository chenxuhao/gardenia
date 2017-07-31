// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include <omp.h>
#include <algorithm>
#include "timer.h"
#include "omp_target_config.h"
#define BC_VARIANT "omp_target"

#pragma omp declare target
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

void PBFS(int m, int *row_offsets, int *column_indices, int source, int *path_counts,  int *depths,
	Bitmap &succ, int **depth_index, int &depth_index_size, SlidingQueue<int> &queue) {
	depths[source] = 0;
	path_counts[source] = 1;
	queue.push_back(source);
	//depth_index.push_back(queue.begin());
	depth_index[depth_index_size++] = queue.begin();
	queue.slide_window();
	#pragma omp parallel
	{
		int depth = 0;
		QueueBuffer<int> lqueue(queue);
		while (!queue.empty()) {
			#pragma omp single
			depth_index[depth_index_size++] = queue.begin();
			//depth_index.push_back(queue.begin());
			depth++;
			#pragma omp for schedule(dynamic, 64)
			for (int *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
				int src = *q_iter;
				int row_begin = row_offsets[src];
				int row_end = row_offsets[src + 1];
				for (int offset = row_begin; offset < row_end; offset ++) {
					int dst = column_indices[offset];
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
	//depth_index.push_back(queue.begin());
	depth_index[depth_index_size++] = queue.begin();
}
#pragma omp end declare target

void BCSolver(int m, int nnz, int source, int *row_offsets, int *column_indices, ScoreT *scores) {
	int num_iters = 1;
	Timer t;
	int *path_counts = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++) path_counts[i] = 0;
	int *depths = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++) depths[i] = -1;
	ScoreT *deltas = (ScoreT *)malloc(m * sizeof(ScoreT));
	for (int i = 0; i < m; i++) deltas[i] = 0;
	warm_up();
	double t1, t2;
	
#pragma omp target data device(0) map(tofrom:scores[0:m]) map(to:depths[0:m],path_counts[0:m]) map(to:deltas[0:m]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz]) map(to:m,nnz,source)
{
	for (int iter = 0; iter < num_iters; iter++) {
		for (int i = 0; i < m; i++) path_counts[i] = 0;
		for (int i = 0; i < m; i++) depths[i] = -1;
		for (int i = 0; i < m; i++) deltas[i] = 0;
		#pragma omp target device(0)
		{
		Bitmap succ(nnz);
		SlidingQueue<int> queue(m);
		int **depth_index = (int **) malloc(sizeof(int *)*m);
		int depth_index_size = 0;
		queue.reset();
		succ.reset();
		t1 = omp_get_wtime();
		PBFS(m, row_offsets, column_indices, source, path_counts, depths, succ, depth_index, depth_index_size, queue);
		//printf("depth_index_size = %d\n", depth_index_size);
		for (int d = depth_index_size-2; d >= 0; d --) {
			#pragma omp parallel for schedule(dynamic, 64)
			for (int *it = depth_index[d]; it < depth_index[d+1]; it++) {
				int src = *it;
				int row_begin = row_offsets[src];
				int row_end = row_offsets[src + 1];
				ScoreT delta_src = deltas[src];
				//#pragma ivdep
				#pragma omp simd reduction(+ : delta_src)
				for (int offset = row_begin; offset < row_end; offset ++) {
					int dst = column_indices[offset];
					if (succ.get_bit(offset)) {
						delta_src += static_cast<ScoreT>(path_counts[src]) /
							static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
					}
				}
				deltas[src] = delta_src;
				scores[src] += deltas[src];
			}
		}
		t2 = omp_get_wtime();
		}
	}
}
	t.Start();
	ScoreT biggest_score = 0;
	#pragma omp parallel for reduction(max : biggest_score)
	for (int n = 0; n < m; n ++)
		biggest_score = max(biggest_score, scores[n]);
	#pragma omp parallel for
	for (int n = 0; n < m; n ++)
		scores[n] = scores[n] / biggest_score;
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs() + 1000*(t2-t1));
	return;
}
