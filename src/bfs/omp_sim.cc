// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include <stdlib.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#ifdef SIM
#include "sim.h"
#endif

#define BFS_VARIANT "omp_base"

void bfs_step(int m, IndexType *row_offsets, IndexType *column_indices, DistT *depth, SlidingQueue<IndexType> &queue) {
#pragma omp parallel
	{
		QueueBuffer<IndexType> lqueue(queue);
#pragma omp for
		for (IndexType *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			IndexType src = *q_iter;
			const IndexType row_begin = row_offsets[src];
			const IndexType row_end = row_offsets[src + 1];
			for (IndexType offset = row_begin; offset < row_end; offset ++) {
				IndexType dst = column_indices[offset];
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

void BFSSolver(int m, int nnz, int source, IndexType *in_row_offsets, IndexType *in_column_indices, IndexType *out_row_offsets, IndexType *out_column_indices, int *in_degree, int *degree, DistT *dist) {
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
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);
	DistT *depth = (DistT *)malloc(m * sizeof(DistT));
	for(int i = 0; i < m; i ++) depth[i] = MYINFINITY;
	depth[source] = 0;
	int iter = 0;
	Timer t;
	t.Start();
#ifdef SIM
	//int *hub_bitmap = (int *) malloc(m * sizeof(int));
	Bitmap hub(m);
	hub.reset();
	int num_hubs = set_hub(m, nnz, in_degree, hub);
	m5_checkpoint(0,0);
#endif
	SlidingQueue<IndexType> queue(m);
	queue.push_back(source);
	queue.slide_window();
#ifdef SIM
	set_addr_bounds(1,(uint64_t)out_row_offsets,(uint64_t)&out_row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)out_column_indices,(uint64_t)&out_column_indices[nnz],8);
	set_addr_bounds(3,(uint64_t)depth,(uint64_t)&depth[m],8);
	set_addr_bounds(6,(uint64_t)hub.start_,(uint64_t)hub.end_,8);
	printf("Begin of ROI\n");
	printf("This graph has %d hub vertices\n", num_hubs);
#endif
	while (!queue.empty()) {
		++ iter;
		printf("iteration=%d, num_frontier=%ld\n", iter, queue.size());
		bfs_step(m, out_row_offsets, out_column_indices, depth, queue);
		queue.slide_window();
	}
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	for(int i = 0; i < m; i ++) dist[i] = depth[i];
	return;
}
