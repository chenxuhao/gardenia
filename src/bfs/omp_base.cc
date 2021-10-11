// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "bfs.h"
#include <omp.h>
#include <stdlib.h>
#include "sim.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

void bfs_step(Graph &g, DistT *depth, SlidingQueue<IndexT> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<IndexT> lqueue(queue);
    #pragma omp for
    for (IndexT *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      IndexT src = *q_iter;
      for (auto dst : g.N(src)) {
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

void BFSSolver(Graph &g, int source, DistT *dist) {
  auto m = g.V();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);
  DistT *depth = (DistT *)malloc(m * sizeof(DistT));
  for (int i = 0; i < m; i ++) depth[i] = MYINFINITY;
  depth[source] = 0;
  int iter = 0;
  Timer t;
  t.Start();
  roi_begin(g, depth);
  SlidingQueue<IndexT> queue(m);
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    //printf("iteration=%d, num_frontier=%ld\n", iter, queue.size());
    bfs_step(g, depth, queue);
    queue.slide_window();
  }
  roi_end();
  t.Stop();
  printf("\titerations = %d.\n", iter);
  printf("\truntime [omp_base] = %f ms.\n", t.Millisecs());
  #pragma omp parallel for
  for (int i = 0; i < m; i ++)
    dist[i] = depth[i];
  return;
}

