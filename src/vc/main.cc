// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include "vc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Vertex Coloring by Xuhao Chen\n");
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetric(0/1)] [symmetrize(0/1)] [reverse(0/1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google 1\n";
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  auto m = g.V();
  auto colors = custom_alloc_global<int>(m);
  for(int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
  VCSolver(g, colors);
  VCVerifier(g, colors);
  return 0;
}
