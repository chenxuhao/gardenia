// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define BFS_VARIANT "omp_base"

void bfs_step(int m, int *row_offsets, int *column_indices, vector<int> &depth, SlidingQueue<int> &queue) {
#pragma omp parallel
	{
		QueueBuffer<int> lqueue(queue);
#pragma omp for
		for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			int src = *q_iter;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				//int curr_val = parent[dst];
				int curr_val = depth[dst];
				if (curr_val == MYINFINITY) { // not visited
					//if (compare_and_swap(parent[dst], curr_val, src)) {
					if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
						lqueue.push_back(dst);
					}
				}
			}
		}
		lqueue.flush();
	}
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, DistT *dist) {
	//omp_set_num_threads(12);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);
	Timer t;
	vector<int> depth(m, MYINFINITY);
	depth[source] = 0;
	SlidingQueue<int> queue(m);
	queue.push_back(source);
	queue.slide_window();
	int iter = 0;
	t.Start();
	while (!queue.empty()) {
		++ iter;
		bfs_step(m, out_row_offsets, out_column_indices, depth, queue);
		queue.slide_window();
		//printf("iteration=%d, num_frontier=%ld\n", iter, queue.size());
	}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	for(int i = 0; i < m; i ++) dist[i] = depth[i];
	return;
}
