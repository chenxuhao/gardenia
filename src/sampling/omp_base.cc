// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "timer.h"
#include "graph.hh"

void Sampling(Graph &g, int *clusters) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  Timer t;
  t.Start();
  #pragma omp parallel for 
  for (VertexId u = 0; u < g.V(); u ++) {
    for (auto v : g.N(u)) {
    } 
  }
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

