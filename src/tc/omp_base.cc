// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"
#include "sim.h"
#include "automine_omp.h"

void TCSolver(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP TC solver (%d threads) ...\n", num_threads);
  Timer t;
  t.Start();
  roi_begin(g, NULL);
#ifdef USE_MERGE
  automine_tc(g, total);
#else
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (VertexId u = 0; u < g.V(); u ++) {
    auto yu = g.N(u);
    for (auto v : yu) {
      counter += (uint64_t)intersection_num(yu, g.N(v));
    } 
  }
  total = counter;
#endif
  roi_end();
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

