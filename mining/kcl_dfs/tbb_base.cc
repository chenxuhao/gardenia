// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/combinable.h>
#include "kcl.h"
#include "timer.h"
#include "emb_list.h"
#include "automine_tbb.h" // ad-hoc automine
#include "ccode_recurse.h" // recursive ccode

void ccode_kclique(Graph &g, unsigned k, uint64_t &total,
                   std::vector<EmbList> &emb_lists, 
                   std::vector<std::vector<uint8_t>> &ccodes) {
  tbb::combinable<uint64_t> sum;
  tbb::parallel_for(tbb::blocked_range<vidType>(0, g.size(),1),
              [&g, &k, &sum, &emb_lists, &ccodes](tbb::blocked_range<vidType> &r) {
    auto tid = tbb::task_arena::current_thread_index();
    uint64_t counter = 0;
    auto &local_ccodes = ccodes[tid];
    auto &emb_list = emb_lists[tid];
    for(vidType v0 = r.begin(); v0 != r.end(); v0++) {
      unsigned level = 0;
      emb_list.set_size(level+1, 0);
      auto begin = g.edge_begin(v0);
      auto end = g.edge_end(v0);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        local_ccodes[dst] = level+1;
        emb_list.add_emb(level+1, dst);
      }
      extend_clique(level+1, k, g, emb_list, local_ccodes, counter);
      reset_ccodes(level, emb_list, local_ccodes);
    }
    sum.local() += counter;
  });
  total = sum.combine(std::plus<uint64_t>());
}

void KCLSolver(Graph &g, unsigned k, uint64_t &total) {
  int num_threads = 1;
  const char* cnthreads = getenv("TBB_NUM_THREADS");
  if (cnthreads) num_threads = std::max(1, atoi(cnthreads));
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
  printf("TBB %d-clique listing solver using DFS (%d threads)\n", k, num_threads);

  std::vector<EmbList> emb_lists(num_threads);
  std::vector<std::vector<uint8_t>> ccodes(num_threads);
  auto max_degree = g.get_max_degree();
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_ccodes = ccodes[tid];
    local_ccodes.resize(g.size()); // the connectivity code
    std::fill(local_ccodes.begin(), local_ccodes.end(), 0);
    auto &emb_list = emb_lists[tid];
    emb_list.init(k, max_degree);
  }

  Timer t;
  t.Start();
  uint64_t sum = 0;
  //automine_kclique(g, k, sum);
  ccode_kclique(g, k, sum, emb_lists, ccodes);
  total = sum;
  t.Stop();
  printf("Number of %d-cliques: %ld\n", k, total);
  printf("runtime [%s] = %f sec\n", "tbb_base", t.Seconds());
  return;
}

