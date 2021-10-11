// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "sgl.h"

int main(int argc, char **argv) {
  if(argc != 3) {
    std::cerr << "usage: " << argv[0] << " <graph prefix> <pattern>\n";
    exit(1);
  }
  std::cout << "Subgraph Listing/Counting (undirected graph only)\n";
  Graph g(argv[1]);
  Pattern patt(argv[2]);
  std::cout << "Pattern: " << patt.get_name() << "\n";
  //for (int vid = 0; vid < 10; vid ++) 
  //  std::cout << "vid " << vid << ": " << g.out_degree(vid) << "\n";
  uint64_t h_total = 0;
  auto m = g.num_vertices();
  auto nnz = g.num_edges();
  std::cout << "|V| " << m << " |E| " << nnz << "\n";
  SglSolver(g, patt, h_total);
  std::cout << "total_num = " << h_total << "\n";
  return 0;
}

