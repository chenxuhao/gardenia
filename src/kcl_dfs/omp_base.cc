// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include <omp.h>
#include "defines.h"
#include "kcl.h"
// #include "sim.h"
#include "cmap.h"
#include "profiler.h"
#include "emb_list.h"
#include "ccode_omp.h"
#include "automine_omp.h"

static inline void reset_cmap(unsigned level, EmbList &emb_list, cmap8_t &cmap) {
  for (VertexId id = 0; id < emb_list.size(level+1); id++) {
    auto v = emb_list.get_vertex(level+1, id);
    cmap.set(v, level);
  }
}

void extend_clique(unsigned level, unsigned k, Graph &g, 
                   EmbList &emb_list, cmap8_t &cmap,
                   uint64_t &counter) {
  if (level == k - 2) {
    uint64_t local_counter = 0;
    for (VertexId emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
      auto v     = emb_list.get_vertex(level, emb_id);
      auto begin = g.edge_begin(v);
      auto end = g.edge_end(v);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        #if USE_DAG == 0
        if (dst >= v) break;
        #endif
        // if (is_clique(level, dst, cmap))
        //   local_counter ++;
        local_counter += is_clique(level, dst, cmap);
      }
    }
    counter += local_counter;
    return;
  }
  for (VertexId emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
    auto v     = emb_list.get_vertex(level, emb_id);
    emb_list.set_size(level+1, 0);
    auto begin = g.edge_begin(v);
    auto end = g.edge_end(v);
    for (auto e = begin; e < end; e++) {
      auto dst = g.getEdgeDst(e);
      #if USE_DAG == 0
      if (dst >= v) break;
      #endif
      if (is_clique(level, dst, cmap)) {
        emb_list.add_emb(level+1, dst);
        cmap.set(dst, level+1);
      }
    }
    extend_clique(level+1, k, g, emb_list, cmap, counter);
    reset_cmap(level, emb_list, cmap);
  }
}
#if 0 
void cmap_kclique(Graph &g, unsigned k, uint64_t &total, 
                  std::vector<cmap8_t> &cmaps,
                  std::vector<EmbList> &emb_lists) {
  if (k == 4) {
    cmap_4clique(g, total, cmaps, emb_lists);
  } else if (k == 5) {
    cmap_5clique(g, total, cmaps, emb_lists);
  } else {
    std::cout << "Not implemented yet";
  }
}
#else
void cmap_kclique(Graph &g, unsigned k, uint64_t &total, 
                   std::vector<cmap8_t> &cmaps,
                   std::vector<EmbList> &emb_lists) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (VertexID v = 0; v < g.size(); v ++) {
    auto tid = omp_get_thread_num();
    auto &cmap = cmaps[tid];
    auto &emb_list = emb_lists[tid];
    unsigned level = 0;
    emb_list.set_size(level+1, 0);
    auto begin = g.edge_begin(v);
    auto end = g.edge_end(v);
    for (auto e = begin; e < end; e++) {
      auto dst = g.getEdgeDst(e);
      #if USE_DAG == 0
      if (dst >= v) break;
      #endif
      cmap.set(dst, level+1);
      emb_list.add_emb(level+1, dst);
    }
    extend_clique(level+1, k, g, emb_list, cmap, counter);
    reset_cmap(level, emb_list, cmap);
  }
  total = counter;
}
#endif
void KCLSolver(Graph &g, unsigned k, uint64_t &total, int num_threads) {
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP %d-clique listing solver using DFS (%d threads)\n",
         k, num_threads);
  std::vector<EmbList> emb_lists(num_threads);
  std::vector<cmap8_t> cmaps(num_threads);
  auto max_degree = g.get_max_degree();
  for (int tid = 0; tid < num_threads; tid ++) {
#ifdef IDENT_CMAP
    cmaps[tid].init(g.size());
#else
    cmaps[tid].init(max_degree);
#endif
    emb_lists[tid].init(k, max_degree);
  }

  Timer t;
  t.Start();
  double start_time = omp_get_wtime();
  // roi_begin();
  //profiler::profilePapi([&]() {
#ifdef USE_CMAP
  cmap_kclique(g, k, total, cmaps);
  // cmap_kclique(g, k, total, cmaps, emb_lists);
#else
  automine_kclique(g, k, total);
#endif
  //}, "k-clique-omp");
  // roi_end();
  double run_time = omp_get_wtime() - start_time;
  t.Stop();
  std::cout << "Number of " << k << "-cliques: " << total << "\n";
  std::cout << "runtime [omp_base] = " << run_time << " sec\n";
#ifdef CUCKOO_CMAP
  print_cuckoo_stats(cmaps);
#endif
  return;
}

