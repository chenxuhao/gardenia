// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define BFS_VARIANT "omp_beamer"

//int64_t BUStep(int m, int *row_offsets, int *column_indices, vector<int> &parent, Bitmap &front, Bitmap &next) {
int64_t BUStep(int m, int *row_offsets, int *column_indices, vector<int> &depth, Bitmap &front, Bitmap &next) {
	int64_t awake_count = 0;
	next.reset();
#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
	for (int src = 0; src < m; src ++) {
		//if (parent[src] < 0) {
		if (depth[src] < 0) { // not visited
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				if (front.get_bit(dst)) {
					//parent[src] = dst;
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

//int64_t TDStep(int m, int *row_offsets, int *column_indices, vector<int> &parent, SlidingQueue<int> &queue) {
int64_t TDStep(int m, int *row_offsets, int *column_indices, vector<int> &depth, SlidingQueue<int> &queue) {
	int64_t scout_count = 0;
#pragma omp parallel
	{
		QueueBuffer<int> lqueue(queue);
#pragma omp for reduction(+ : scout_count)
		for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			int src = *q_iter;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				//int curr_val = parent[dst];
				int curr_val = depth[dst];
				if (curr_val < 0) { // not visited
					//if (compare_and_swap(parent[dst], curr_val, src)) {
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
#pragma omp parallel for
	for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
		int u = *q_iter;
		bm.set_bit_atomic(u);
	}
}

void BitmapToQueue(int m, const Bitmap &bm, SlidingQueue<int> &queue) {
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

vector<int> InitParent(int m, int *degree) {
	vector<int> parent(m);
#pragma omp parallel for
	for (int n = 0; n < m; n++)
		parent[n] = degree[n] != 0 ? -degree[n] : -1;
	return parent;
}

vector<int> InitDepth(int m, int *degree) {
	vector<int> depth(m);
#pragma omp parallel for
	for (int n = 0; n < m; n++)
		depth[n] = degree[n] != 0 ? -degree[n] : -1;
	return depth;
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, DistT *dist) {
	//omp_set_num_threads(12);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);

	int alpha = 15, beta = 18;
	Timer t;
	//vector<int> parent = InitParent(m, degree);
	//parent[source] = source;
	//vector<int> depth(m, MYINFINITY);
	vector<int> depth = InitDepth(m, degree);
	depth[source] = 0;
	SlidingQueue<int> queue(m);
	queue.push_back(source);
	queue.slide_window();
	Bitmap curr(m);
	curr.reset();
	Bitmap front(m);
	front.reset();
	int64_t edges_to_check = nnz;
	int64_t scout_count = degree[source];
	int iter = 0;
	t.Start();
	while (!queue.empty()) {
		if (scout_count > edges_to_check / alpha) {
			int64_t awake_count, old_awake_count;
			QueueToBitmap(queue, front);
			awake_count = queue.size();
			queue.slide_window();
			do {
				++ iter;
				old_awake_count = awake_count;
				//awake_count = BUStep(m, in_row_offsets, in_column_indices, parent, front, curr);
				awake_count = BUStep(m, in_row_offsets, in_column_indices, depth, front, curr);
				front.swap(curr);
				//printf("BU: ");
				//printf("iteration=%d, num_frontier=%ld\n", iter, awake_count);
			} while ((awake_count >= old_awake_count) ||
					(awake_count > m / beta));
			BitmapToQueue(m, front, queue);
			scout_count = 1;
		} else {
			++ iter;
			edges_to_check -= scout_count;
			//scout_count = TDStep(m, out_row_offsets, out_column_indices, parent, queue);
			scout_count = TDStep(m, out_row_offsets, out_column_indices, depth, queue);
			queue.slide_window();
			//printf("TD: (scout_count=%ld) ", scout_count);
			//printf("TD: iteration=%d, num_frontier=%ld\n", iter, queue.size());
		}
	}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	for(int i = 0; i < m; i ++) if(depth[i]>=0) dist[i] = depth[i]; else dist[i] = MYINFINITY;
	return;
}
