// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "sssp.h"
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include "sim.h"
#include "platform_atomics.h"

//[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
//    algorithm." Journal of Algorithms, 49(1):114–152, 2003.
void SSSPSolver(Graph &g, int source, DistT *weight, DistT *dist, int delta) {
  int num_threads = 1;
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP SSSP solver (%d threads) ...\n", num_threads);
  Timer t;
  dist[source] = 0;
  IndexT *frontier = (IndexT *)malloc(g.E()*sizeof(IndexT));
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kDistInf};
  size_t frontier_tails[2] = {1, 0}; 
  frontier[0] = source;

  t.Start();
  roi_begin(g, dist, frontier, weight);
#pragma omp parallel
  {
    vector<vector<IndexT> > local_bins(0);
    int iter = 0;
    while (static_cast<DistT>(shared_indexes[iter&1]) != kDistInf) {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
#pragma omp for nowait schedule(dynamic, 64)
      for (size_t i = 0; i < curr_frontier_tail; i ++) {
        IndexT src = frontier[i];
        if (dist[src] >= delta * static_cast<DistT>(curr_bin_index)) {
          auto offset = g.edge_begin(src);
          for (auto dst : g.N(src)) {
            DistT old_dist = dist[dst];
            DistT new_dist = dist[src] + weight[offset++];
            if (new_dist < old_dist) {
              bool changed_dist = true;
              while (!compare_and_swap(dist[dst], old_dist, new_dist)) {
                old_dist = dist[dst];
                if (old_dist <= new_dist) {
                  changed_dist = false;
                  break;
                }
              }
              if (changed_dist) {
                size_t dest_bin = new_dist/delta;
                if (dest_bin >= local_bins.size()) {
                  local_bins.resize(dest_bin+1);
                }
                local_bins[dest_bin].push_back(dst);
              }
            }
          }
        }
      }
      for (size_t i = curr_bin_index; i < local_bins.size(); i ++) {
        if (!local_bins[i].empty()) {
#pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
#pragma omp barrier
#pragma omp single nowait
      {
        curr_bin_index = kDistInf;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
            local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
            local_bins[next_bin_index].end(), frontier + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
#pragma omp barrier
    }
  }
  roi_end();
  t.Stop();
  //printf("\titerations = %d.\n", iter);
  printf("\truntime [omp_base] = %f ms.\n", t.Millisecs());
  return;
}

