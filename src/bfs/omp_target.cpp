// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define BFS_VARIANT "omp_target"

#pragma omp declare target
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

int64_t BUStep(int m, int *row_offsets, int *column_indices, int *depth, Bitmap &front, Bitmap &next) {
	int64_t awake_count = 0;
	next.reset();
	//#pragma omp target device(0)
	#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
	for (int src = 0; src < m; src ++) {
		if (depth[src] < 0) { // not visited
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				if (front.get_bit(dst)) {
					depth[src] = depth[dst] + 1;
					awake_count++;
					next.set_bit(src);
					break;
				}
			}
		}
	}
	return awake_count;
}

int64_t TDStep(int m, int *row_offsets, int *column_indices, int *depth, SlidingQueue<int> &queue) {
	int64_t scout_count = 0;
	//#pragma omp target device(0)
	#pragma omp parallel
	{
		QueueBuffer<int> lqueue(queue);
		#pragma omp for reduction(+ : scout_count)
		for (int *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			int src = *q_iter;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				int curr_val = depth[dst];
				if (curr_val < 0) { // not visited
					if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
						lqueue.push_back(dst);
						scout_count += -curr_val;
					}
				}
			}
		}
		lqueue.flush();
	}
	return scout_count;
}

void QueueToBitmap(const SlidingQueue<int> &queue, Bitmap &bm) {
	//#pragma omp target device(0)
	#pragma omp parallel for
	for (int *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
		int u = *q_iter;
		bm.set_bit_atomic(u);
	}
}

void BitmapToQueue(int m, const Bitmap &bm, SlidingQueue<int> &queue) {
	//#pragma omp target device(0)
	#pragma omp parallel
	{
		QueueBuffer<int> lqueue(queue);
		#pragma omp for
		for (int n = 0; n < m; n++)
			if (bm.get_bit(n))
				lqueue.push_back(n);
		lqueue.flush();
	}
	queue.slide_window();
}
#pragma omp end declare target

int * InitDepth(int m, int *degree) {
	int *depth = (int *) malloc(m*sizeof(int));
	#pragma omp parallel for
	for (int n = 0; n < m; n++)
		depth[n] = degree[n] != 0 ? -degree[n] : -1;
	return depth;
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *in_degree, int *degree, DistT *dist) {
	const int alpha = 15, beta = 18;
	Timer t;
	int *depth = InitDepth(m, degree);
	depth[source] = 0;
	warm_up();
	double t1, t2;
	t.Start();
	int iter = 0;
#pragma omp target data device(0) map(tofrom:depth[0:m]) map(to:in_row_offsets[0:(m+1)],out_row_offsets[0:(m+1)]) map(to:in_column_indices[0:nnz],out_column_indices[0:nnz]) map(to:m,nnz,source) map(tofrom:iter)
{

	#pragma omp target device(0)
	{
	SlidingQueue<int> queue(m);
	queue.push_back(source);
	queue.slide_window();
	Bitmap curr(m);
	curr.reset();
	Bitmap front(m);
	front.reset();
	int64_t edges_to_check = nnz;
	int64_t scout_count = degree[source];
	
	t1 = omp_get_wtime();
	while (!queue.empty()) {
		if (scout_count > edges_to_check / alpha) {
			int64_t awake_count, old_awake_count;
			QueueToBitmap(queue, front);
			awake_count = queue.size();
			queue.slide_window();
			do {
				++ iter;
				old_awake_count = awake_count;
				awake_count = BUStep(m, in_row_offsets, in_column_indices, depth, front, curr);
				front.swap(curr);
			} while ((awake_count >= old_awake_count) ||
					(awake_count > m / beta));
			BitmapToQueue(m, front, queue);
			scout_count = 1;
		} else {
			++ iter;
			edges_to_check -= scout_count;
			scout_count = TDStep(m, out_row_offsets, out_column_indices, depth, queue);
			queue.slide_window();
		}
	}
	t2 = omp_get_wtime();
	}
}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	//printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, 1000*(t2-t1));
	for(int i = 0; i < m; i ++) if(depth[i]>=0) dist[i] = depth[i]; else dist[i] = MYINFINITY;
	return;
}
