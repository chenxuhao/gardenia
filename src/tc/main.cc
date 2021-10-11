// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }
  std::cout << "Triangle Count (for undirected graphs only)\n";
  Graph g(argv[1]); // use DAG
  g.orientation();
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  uint64_t total = 0;
  TCSolver(g, total);
  std::cout << "total_num_triangles = " << total << "\n";
  //TCVerifier(g, total);
  return 0;
}

