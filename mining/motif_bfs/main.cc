// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "motif.h"
#include "builder.h"
#include "mgraph_reader.h"
#include "command_line.h"
static int num_patterns[3] = {2, 6, 21};

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
	read_graph(g, filetype, filename, false);//use symmetric graph
	int npatterns = num_patterns[k-3];
	std::cout << k << "-motif has " << npatterns << " patterns in total\n";
	std::vector<AccType> accumulators(npatterns);
	for (int i = 0; i < npatterns; i++) accumulators[i] = 0;
	
	int m = g.num_vertices();
	int nnz = g.num_edges();
	printf("After cleaning: num_vertices %d num_edges %d\n", m, nnz);
	//g.print_graph();
	MotifSolver(g, k, accumulators);
	return 0;
}

