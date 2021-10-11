void automine_tc(Graph &g, uint64_t &total) {
  uint64_t counter = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:counter)
  for (VertexId v0 = 0; v0 < g.V(); v0++) {
    VertexSet y0 = g.N(v0);
    for (auto v1 : y0) {
      VertexSet y1 = g.N(v1);
      counter += intersection_num(y0, y1);
    }
  }
  total = counter;
}
