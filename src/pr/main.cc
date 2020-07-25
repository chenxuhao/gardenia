// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "pr.h"

int main(int argc, char *argv[]) {
	std::cout << "PageRank by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetrize(0/1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google\n";
    exit(1);
  }
  bool symmetrize = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  Graph g(argv[2], argv[1], symmetrize, 1);
  auto m = g.V();
  const ScoreT init_score = 1.0f / m;
  std::vector<ScoreT> scores(m, init_score);
  PRSolver(g, &scores[0]);
  PRVerifier(g, &scores[0], EPSILON);
  return 0;
}
