// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "bc.h"

int main(int argc, char *argv[]) {
  std::cout << "Betweenness Centrality by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetrize(0/1)] [reverse(0/1)] [source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google\n";
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  int source = 0;
  if (argc == 6) source = atoi(argv[5]);
  auto m = g.V();
  std::vector<ScoreT> scores(m, 0);
  BCSolver(g, source, &scores[0]);
  BCVerifier(g, source, 1, &scores[0]);
  return 0;
}
