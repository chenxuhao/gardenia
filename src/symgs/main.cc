// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "symgs.h"
#include "../vc/vc.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  std::cout << "Symmetric Gauss-Seidel smoother by Xuhao Chen\n"
            << "Note: this application uses incoming edge-list, "
            << "which requires reverse (transposed) graph\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetric(0/1)] [symmetrize(0/1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google 0\n";
    exit(1);
  }
  bool symmetrize = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  Graph g(argv[2], argv[1], symmetrize, true);
  auto m = g.V();
  auto nnz = g.E();

  auto h_x = custom_alloc_global<ValueT>(m);
  auto h_b = custom_alloc_global<ValueT>(m);
  auto x_host = custom_alloc_global<ValueT>(m);
  auto weights = custom_alloc_global<ValueT>(nnz);

  // fill matrix with random values: some matrices have extreme values,
  // which makes correctness testing difficult, especially in single precision
  srand(13);
  //for(int i = 0; i < nnz; i++) {
  //weights[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[]
  //weights[i] = rand() / (RAND_MAX + 1.0); // Ax[]
  //}
  for(int i = 0; i < m; i++) {
    h_x[i] = rand() / (RAND_MAX + 1.0);
    x_host[i] = h_x[i];
  }
  for(int i = 0; i < m; i++) 
    h_b[i] = rand() / (RAND_MAX + 1.0);

  // identify parallelism using vertex coloring
  int *ordering = (int *)malloc(m * sizeof(int));
  //for(int i = 0; i < m; i++) ordering[i] = i;
  thrust::sequence(ordering, ordering+m);
  int *colors = (int *)malloc(m * sizeof(int));
  for(int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
  int num_colors = VCSolver(g, colors);
  thrust::sort_by_key(colors, colors+m, ordering);
  int *temp = (int *)malloc((num_colors+1) * sizeof(int));
  thrust::reduce_by_key(colors, colors+m,
      thrust::constant_iterator<int>(1), 
      thrust::make_discard_iterator(), temp);
  thrust::exclusive_scan(temp, temp+num_colors+1, temp, 0);
  std::vector<int> color_offsets(num_colors+1);
  for(size_t i = 0; i < color_offsets.size(); i ++) color_offsets[i] = temp[i];

  SymGSSolver(g, ordering, weights, h_x, h_b, color_offsets);
  SymGSVerifier(g, ordering, weights, h_x, x_host, h_b, color_offsets);
  return 0;
}
