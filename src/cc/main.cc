// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "cc.h"
#include "csr_graph.h"

int main(int argc, char *argv[]) {
  printf("Connected Component by Xuhao Chen\n");
  if (argc < 3) {
    printf("Usage: %s <filetype> <graph> [symmetrize(0/1)] [reverse(0/1)]\n", argv[0]);
    printf("Example: %s mtx web-Google 1 1\n", argv[0]);
    exit(1);
  }
  Graph g(argv[2], argv[1], atoi(argv[3]), atoi(argv[4]));
  auto m = g.V();
  //CompT *h_comp = (CompT *)aligned_alloc(PAGE_SIZE, m * sizeof(CompT));
  CompT *h_comp = (CompT *)malloc(m * sizeof(CompT));
  for (int i = 0; i < m; i++) h_comp[i] = i;
  CCSolver(g, h_comp);
  CCVerifier(g, h_comp);
  return 0;
}
