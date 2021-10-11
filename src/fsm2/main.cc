// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include "fsm.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Frequent Subgraph Mining by Xuhao Chen (only for undirected graphs)\n");
	if(argc != 5 && argc != 4) {
		std::cerr << "usage: " << argv[0] << " <filetype> <filename> <minimal-support> [<output-filename>]" << std::endl;
		return 1;
	}
	string filetype = argv[1];
	string filename = argv[2];
	unsigned minsup = atoi(argv[3]);
	unsigned k = atoi(argv[4]);
	string output_filename;
	if(argc == 6) output_filename = argv[4];
	if(minsup < 1) {
		cerr << "error: minsup < 1" << endl;
		return 3;
	}
	printf("minsup = %d\n", minsup);

	Graph graph;
	int m, n, nnz;
	int max = 10;
	if (filetype == "mtx") {
		IndexT *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
		WeightT *h_weights = NULL;
		read_graph(argc, &argv[1], m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weights, true);
		int *labels = (int *)malloc(m * sizeof(int));
		if (m > 10000) max = 100;
		for (int i = 0; i < m; i ++) {
			labels[i] = rand() % max + 1;
			if(i<8) printf("labels[%d]=%d\n", i, labels[i]);
		}
		graph.read_csr(m, nnz, h_row_offsets, h_column_indices, labels, NULL);
	} else if (filetype == "txt") {
		printf("Reading TXT file: %s\n", filename.c_str());
		std::ifstream in;
		in.open(filename.c_str(), std::ios::in);
		graph.read_txt(in);
		in.close();
	} else if (filetype == "adj") {
		printf("Reading ADJ file: %s\n", filename.c_str());
		std::ifstream in;
		in.open(filename.c_str(), std::ios::in);
		graph.read_adj(in);
		in.close();
	} else { printf("Unkown file format\n"); exit(1); }
	printf("num_vertices = %d, num_edges = %d\n", graph.vertex_size(), graph.edge_size());
	size_t total = 0;
	FSMSolver(graph, minsup, k, total);
	FSMVerifier(graph, minsup, total);
	return 0;
}
