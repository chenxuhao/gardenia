// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "tc.h"
//#include "builder.h"
#include "mgraph_reader.h"
#include "command_line.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <filetype> <graph>\n", argv[0]);
    printf("Example: %s mtx web-Google.mtx\n", argv[0]);
    exit(1);
  }
  CLApp cli(argc, argv, "triangle count");
  printf("Triangle Count by Xuhao Chen (for undirected graphs only)\n");
  Graph g;
  std::string filetype = argv[1];
  std::string filename = argv[2];
  read_graph(g, filetype, filename);
  //if (!cli.ParseArgs()) return -1;
  //Builder b(cli);
  //b.MakeGraph(g);
  uint64_t h_total = 0;
  int m = g.num_vertices();
  int nnz = g.num_edges();
  printf("After cleaning: num_vertices %d num_edges %d\n", m, nnz);

  TCSolver(g, h_total);
  std::cout << "total_num_triangles = " << h_total << "\n";
  //if (cli.do_verify()) TCVerifier(g, h_total);
  return 0;
}

