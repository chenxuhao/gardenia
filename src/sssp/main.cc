// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "sssp.h"

int main(int argc, char *argv[]) {
	std::cout << "Single Source Shortest Path by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetrize(0/1)] [reverse(0/1)] [source_id(0)] [delta(1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google 0 1\n";
    exit(1);
  }
	int delta = 1;
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  int source = 0;
  if (argc > 5) source = atoi(argv[5]);
  if (argc > 6) delta = atoi(argv[6]);
	//printf("Delta: %d\n", delta);
  auto m = g.V();
  auto nnz = g.E();
  std::vector<DistT> distances(m, kDistInf);
	std::vector<DistT> wt(nnz, DistT(1));
  SSSPSolver(g, source, &wt[0], &distances[0], delta);
  SSSPVerifier(g, source, &wt[0], &distances[0]);
  return 0;
}

