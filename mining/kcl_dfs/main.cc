// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include <omp.h>
#include "kcl.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <graph> <k> [thread-num]\n", argv[0]);
    exit(1);
  }
  std::cout << "k-clique listing with undirected graphs\n";
  if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  Graph g(argv[1], USE_DAG); // use DAG

  unsigned k = atoi(argv[2]);
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";

  int num_threads = omp_get_max_threads();
  if (argc == 4) {
    num_threads = atoi(argv[3]);
  }

  uint64_t total = 0;
  KCLSolver(g, k, total, num_threads);
  //KCLVerifier(g, k, total);
  return 0;
}
