#pragma once

static inline void print_graph(Graph& g) {
  for (size_t u = 0; u < g.size(); u++) {
    std::cout << "vertex " << u
              << ": degree = " << g.get_degree(u) 
              << " edgelist = [ ";
    //for (auto e : graph.edges(n))
    for (auto v : g.N(u))
      std::cout << v << " ";
    std::cout << "]" << std::endl;
  }
}

