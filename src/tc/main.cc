// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }
  std::cout << "Triangle Counting\n";
  if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  Graph g(argv[1], USE_DAG); // use DAG
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  uint64_t total = 0;
  TCSolver(g, total);
  std::cout << "total_num_triangles = " << total << "\n";
  //TCVerifier(g, total);
  return 0;
}

