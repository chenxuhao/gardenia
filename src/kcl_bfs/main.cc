// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "kcl.h"
#include "builder.h"
#include "mgraph_reader.h"
#include "command_line.h"

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s <filetype> <filename> [max_size(3)]\n", argv[0]);
		exit(1);
	} 
	std::string filetype = argv[1];
	std::string filename = argv[2];
	unsigned k = 3;
	if (argc == 4) k = atoi(argv[3]);
	printf("k = %d\n", k);
	
	Graph g;
	read_graph(g, filetype, filename);
	AccType total = 0;
	int m = g.num_vertices();
	int nnz = g.num_edges();
	printf("After cleaning: num_vertices %d num_edges %d\n", m, nnz);
	//g.print_graph();
	KclSolver(g, k, total);
	return 0;
}

