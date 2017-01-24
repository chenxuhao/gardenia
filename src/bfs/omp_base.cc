#include "bfs.h"
#include <omp.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define BFS_VARIANT "openmp"

int64_t BUStep(int m, int *row_offsets, int *column_indices, vector<int> &parent, Bitmap &front, Bitmap &next) {
	int64_t awake_count = 0;
	next.reset();
#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
	for (int src = 0; src < m; src ++) {
		if (parent[src] < 0) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				if (front.get_bit(dst)) {
					parent[src] = dst;
					awake_count++;
					next.set_bit(src);
					break;
				}
			}
		}
	}
	return awake_count;
}

int64_t TDStep(int m, int *row_offsets, int *column_indices, vector<int> &parent, SlidingQueue<int> &queue) {
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
				int curr_val = parent[dst];
				if (curr_val < 0) {
					if (compare_and_swap(parent[dst], curr_val, src)) {
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

void BFSSolver(int m, int nnz, int *row_offsets, int *column_indices, int *degree, DistT *dist) {
	int source = 0;
	int alpha = 15, beta = 18;
	Timer t;
	vector<int> parent = InitParent(m, degree);
	parent[source] = source;
	SlidingQueue<int> queue(m);
	queue.push_back(source);
	queue.slide_window();
	Bitmap curr(m);
	curr.reset();
	Bitmap front(m);
	front.reset();
	int64_t edges_to_check = nnz;
	int64_t scout_count = degree[source];
	t.Start();
	while (!queue.empty()) {
		if (scout_count > edges_to_check / alpha) {
			int64_t awake_count, old_awake_count;
			//TIME_OP(t, QueueToBitmap(queue, front));
			awake_count = queue.size();
			queue.slide_window();
			do {
				old_awake_count = awake_count;
				awake_count = BUStep(m, row_offsets, column_indices, parent, front, curr);
				front.swap(curr);
			} while ((awake_count >= old_awake_count) ||
					(awake_count > m / beta));
			TIME_OP(t, BitmapToQueue(m, front, queue));
			scout_count = 1;
		} else {
			edges_to_check -= scout_count;
			scout_count = TDStep(m, row_offsets, column_indices, parent, queue);
			queue.slide_window();
		}
	}
	t.Stop();
	return;
}
