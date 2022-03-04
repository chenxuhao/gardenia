// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include "fsm.h"
//#include "graph_io.h"
#include "mgraph.h"

int main(int argc, char **argv) {
	if(argc < 5) {
		std::cerr << "usage: " << argv[0] << " <filetype> <filename> <minimal-support> <max-size> [<output-filename>]" << std::endl;
		return 1;
	}
	string filetype = argv[1];
	string filename = argv[2];
	unsigned minsup = atoi(argv[3]);
	unsigned k = atoi(argv[4]);
	string output_filename;
	if(argc == 5) output_filename = argv[4];
	if(minsup < 1) {
		cerr << "error: minsup < 1" << endl;
		return 3;
	}
	printf("minsup = %d\n", minsup);
	printf("k = %d\n", k);

	MGraph g(true);
	std::ifstream in;
	in.open(filename.c_str(), std::ios::in);
	if (filetype == "mtx") {
		printf("Reading MTX file: %s\n", filename.c_str());
		g.read_mtx(in);
	} else if (filetype == "txt") {
		printf("Reading TXT file: %s\n", filename.c_str());
		g.read_txt(in);
	} else if (filetype == "adj") {
		printf("Reading ADJ file: %s\n", filename.c_str());
		exit(0);
		//g.read_adj(in);
	} else { printf("Unkown file format\n"); exit(1); }
	in.close();
	int m = g.num_vertices();
	int nnz = g.num_edges();
	printf("num_vertices = %d, num_edges = %d\n", m, nnz);
	//g.print_graph();
	long long num_freq_patterns;
	FSMSolver(m, nnz, minsup, k, g.out_rowptr(), g.out_colidx(), g.labels(), &num_freq_patterns);
	//if(cli.do_verify()) FSMVerifier(g, h_total);
	return 0;
}

