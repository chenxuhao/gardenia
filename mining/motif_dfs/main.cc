// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "motif.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <graph> <k>\n", argv[0]);
    exit(1);
  }
  printf("k-motif counting (only for undirected graphs)\n");
  Graph g(argv[1]);
  unsigned k = atoi(argv[2]);
  auto m = g.size();
  auto nnz = g.sizeEdges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  int num_patterns = num_possible_patterns[k];
  std::cout << "num_patterns: " << num_patterns << "\n";
  std::vector<uint64_t> h_total(num_patterns, 0);
  MotifSolver(g, k, h_total);
  return 0;
}

