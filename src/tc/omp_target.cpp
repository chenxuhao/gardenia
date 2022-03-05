// Authors: Xuhao Chen <cxh@cxh.edu>
#include "timer.h"
#include "graph.hh"
#include "omp_target_config.h"

vidType intersect_num(vidType v, vidType u, eidType *vertices, vidType *edges) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = vertices[v+1]-vertices[v];
  vidType u_size = vertices[v+1]-vertices[u];
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b) num++;
  }
  return num;
}

void TCSolver(Graph &g, uint64_t &total) {
  warm_up();
  Timer t;
  t.Start();
  int *h_total = (int *) malloc(sizeof(int));
  eidType* row_offsets = g.out_rowptr();
  vidType* column_indices = g.out_colidx();
  auto m = g.V();
  auto nnz = g.E();
  #pragma omp target data device(0) map(tofrom:h_total[0:1]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz])
  {
    #pragma omp target device(0)
    {
      int total_num = 0;
      #pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 64)
      for (vidType u = 0; u < m; u ++) {
        //auto yu = g.N(u);
        auto row_begin = row_offsets[u];
        auto row_end = row_offsets[u+1]; 
        for (auto offset = row_begin; offset < row_end; ++ offset) {
          vidType v = column_indices[offset];
          total_num += (uint64_t)intersect_num(u, v, row_offsets, column_indices);
        } 
      }
      h_total[0] = total_num;
    }
  }
  total = h_total[0];
  t.Stop();
  std::cout << "runtime [omp_target] = " << t.Seconds() << " sec\n";
  return;
}
