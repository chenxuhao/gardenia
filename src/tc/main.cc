// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <filetype> <graph-prefix> \n", argv[0]);
    printf("Example: %s mtx web-Google\n", argv[0]);
    exit(1);
  }
  std::cout << "Triangle Count (for undirected graphs only)\n";
  Graph g(argv[2], argv[1], 1);
  g.orientation();
  uint64_t total = 0;
  TCSolver(g, total);
  std::cout << "total_num_triangles = " << total << "\n";
  //TCVerifier(g, total);
  return 0;
}

