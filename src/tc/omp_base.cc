// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.hh"
#include "timer.h"

void TCSolver(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "Launching OpenMP TC solver (" << num_threads << " threads) ...\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (VertexId u = 0; u < g.V(); u ++) {
    auto yu = g.N(u);
    for (auto v : yu) {
      counter += (uint64_t)intersection_num(yu, g.N(v));
    } 
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

