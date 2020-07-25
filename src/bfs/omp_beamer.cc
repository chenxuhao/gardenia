// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <vector>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define BFS_VARIANT "omp_beamer"
typedef long int64_t;

//int64_t BUStep(Graph &g, vector<int> &parent, Bitmap &front, Bitmap &next) {
int64_t BUStep(Graph &g, vector<int> &depths, Bitmap &front, Bitmap &next) {
	int64_t awake_count = 0;
	next.reset();
	#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
	for (int dst = 0; dst < g.V(); dst ++) {
		//if (parent[dst] < 0) {
		if (depths[dst] < 0) { // not visited
      for (auto src : g.in_neigh(dst)) {
				if (front.get_bit(src)) {
					//parent[dst] = src;
					depths[dst] = depths[src] + 1;
					awake_count++;
					next.set_bit(dst);
					break;
				}
			}
		}
	}
	return awake_count;
}

//int64_t TDStep(Graph &g, vector<int> &parent, SlidingQueue<int> &queue) {
int64_t TDStep(Graph &g, vector<int> &depths, SlidingQueue<int> &queue) {
	int64_t scout_count = 0;
	#pragma omp parallel
	{
		QueueBuffer<int> lqueue(queue);
		#pragma omp for reduction(+ : scout_count)
		for (int *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
			int src = *q_iter;
      for (auto dst : g.out_neigh(src)) {
				//int curr_val = parent[dst];
				int curr_val = depths[dst];
				if (curr_val < 0) { // not visited
					//if (compare_and_swap(parent[dst], curr_val, src)) {
					if (compare_and_swap(depths[dst], curr_val, depths[src] + 1)) {
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
	for (int *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
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

vector<int> InitParent(int m, int *degrees) {
	vector<int> parent(m);
	#pragma omp parallel for
	for (int n = 0; n < m; n++)
		parent[n] = degrees[n] != 0 ? -degrees[n] : -1;
	return parent;
}

vector<int> InitDepth(int m, int *degrees) {
	vector<int> depths(m);
	#pragma omp parallel for
	for (int n = 0; n < m; n++)
		depths[n] = degrees[n] != 0 ? -degrees[n] : -1;
	return depths;
}

void BFSSolver(Graph &g, int source, DistT *dist) {
  if (!g.has_reverse_graph()) {
    std::cout << "This algorithm requires the reverse graph constructed for directed graph\n";
    std::cout << "Please set reverse to 1 in the command line\n";
    exit(1);
  }
  auto m = g.V();
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);

	int alpha = 15, beta = 18;
  std::vector<VertexId> degrees(m, 0);
  #pragma omp parallel for
	for (VertexId i = 0; i < m; i ++) {
    degrees[i] = g.get_degree(i);
  }
	//vector<int> parent = InitParent(m, degrees);
	//parent[source] = source;
	//vector<int> depths(m, MYINFINITY);
	vector<int> depths = InitDepth(m, &degrees[0]);
	depths[source] = 0;
	SlidingQueue<int> queue(m);
	queue.push_back(source);
	queue.slide_window();
	Bitmap curr(m);
	curr.reset();
	Bitmap front(m);
	front.reset();
	int64_t edges_to_check = g.E();
	int64_t scout_count = degrees[source];
	int iter = 0;

	Timer t;
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
				awake_count = BUStep(g, depths, front, curr);
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
			scout_count = TDStep(g, depths, queue);
			queue.slide_window();
			//printf("TD: (scout_count=%ld) ", scout_count);
			//printf("TD: iteration=%d, num_frontier=%ld\n", iter, queue.size());
		}
	}
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
  #pragma omp parallel for
	for (VertexId i = 0; i < m; i ++) {
    if (depths[i]>=0) dist[i] = depths[i]; 
    else dist[i] = MYINFINITY;
  }
	return;
}

