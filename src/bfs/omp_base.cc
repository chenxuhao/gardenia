// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include <stdlib.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

#define BFS_VARIANT "omp_base"

void bfs_step(int m, IndexT *row_offsets, IndexT *column_indices, DistT *depth, SlidingQueue<IndexT> &queue) {
	#pragma omp parallel
	{
		QueueBuffer<IndexT> lqueue(queue);
		#pragma omp for
		for (IndexT *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			IndexT src = *q_iter;
			const IndexT row_begin = row_offsets[src];
			const IndexT row_end = row_offsets[src+1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = column_indices[offset];
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

void BFSSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *in_degree, int *degree, DistT *dist) {
	//omp_set_num_threads(12);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);
	DistT *depth = (DistT *)malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) depth[i] = MYINFINITY;
	depth[source] = 0;
	int iter = 0;
	Timer t;
	t.Start();
	SlidingQueue<IndexT> queue(m);
	queue.push_back(source);
	queue.slide_window();
	while (!queue.empty()) {
		++ iter;
		printf("iteration=%d, num_frontier=%ld\n", iter, queue.size());
		bfs_step(m, out_row_offsets, out_column_indices, depth, queue);
		queue.slide_window();
	}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	for(int i = 0; i < m; i ++) dist[i] = depth[i];
	return;
}
