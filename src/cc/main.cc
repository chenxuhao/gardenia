// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "cc.h"

int main(int argc, char *argv[]) {
  std::cout << "Connected Component by Xuhao Chen\n";
  if (argc < 3) {
    printf("Usage: %s <filetype> <graph> [symmetrize(0/1)] [reverse(0/1)]\n", argv[0]);
    printf("Example: %s mtx web-Google 1 1\n", argv[0]);
    exit(1);
  }
  Graph g(argv[2], argv[1], atoi(argv[3]), atoi(argv[4]));
  auto m = g.V();
  std::vector<CompT> comp(m);
  for (int i = 0; i < m; i++) comp[i] = i;
  CCSolver(g, &comp[0]);
  CCVerifier(g, &comp[0]);
  return 0;
}
