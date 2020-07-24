// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"

int main(int argc, char *argv[]) {
  printf("Sparse Matrix-Vector Multiplication by Xuhao Chen\n");
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetric(0/1)] [symmetrize(0/1)] [reverse(0/1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google 0 1\n";
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  auto m = g.V();
  auto nnz = g.E();

  //ValueT *h_x = (ValueT *)aligned_alloc(PAGE_SIZE, m * sizeof(ValueT));
  auto h_x = custom_alloc_global<ValueT>(m);
  auto h_y = custom_alloc_global<ValueT>(m);
  auto y_host = custom_alloc_global<ValueT>(m);
  auto in_weights = custom_alloc_global<ValueT>(nnz);
  auto out_weights = custom_alloc_global<ValueT>(nnz);
  srand(13);
  for(size_t i = 0; i < nnz; i++) {
    out_weights[i] = 0.2;//1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[] (-1 ~ 1)
    in_weights[i] = 0.2;//rand() / (RAND_MAX + 1.0); // Ax[] (0 ~ 1)
  }
  for(int i = 0; i < m; i++) {
    //h_x[i] = rand() / (RAND_MAX + 1.0);
    h_x[i] = 0.3;
    h_y[i] = 0.0;//rand() / (RAND_MAX + 1.0);
    y_host[i] = h_y[i];
  }

  SpmvSolver(g, in_weights, h_x, h_y);
  SpmvVerifier(g, in_weights, h_x, y_host, h_y);
  return 0;
}

