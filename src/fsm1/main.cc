// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "fsm.h"
#include "builder.h"
#include "mgraph_reader.h"
#include "command_line.h"

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s <filetype> <filename> [max_size(3)] [min_support(5000)]\n", argv[0]);
		exit(1);
	} 
	std::string filetype = argv[1];
	std::string filename = argv[2];
	unsigned k = 3;
	if (argc > 3) k = atoi(argv[3]);
	unsigned minsup = 5000;
	if (argc > 4) minsup = atoi(argv[4]);
	printf("max_size = %d\n", k);
	printf("min_support = %d\n", minsup);
	
	Graph g;
	int nlabels = read_graph(g, filetype, filename, false, true);//use symmetric graph
	int m = g.num_vertices();
	int nnz = g.num_edges();
	printf("After cleaning: num_vertices %d num_edges %d\n", m, nnz);
	int num_freqent_patterns = 0;
	FsmSolver(g, k, minsup, nlabels, num_freqent_patterns);
	printf("\n\tNumber of frequent patterns: %d\n", num_freqent_patterns);
	return 0;
}

