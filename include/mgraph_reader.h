#include "mgraph.h"
#include "builder.h"

int read_graph(Graph &graph, std::string filetype, std::string filename, bool use_dag = true, bool enable_label = false) {
	MGraph mg(use_dag); // mining graph
	if (filetype == "txt") {
		printf("Reading .lg file: %s\n", filename.c_str());
		mg.read_txt(filename.c_str());
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		mg.read_adj(filename.c_str());
	} else if (filetype == "mtx") {
		printf("Reading .mtx file: %s\n", filename.c_str());
		mg.read_mtx(filename.c_str(), true);
	} else { printf("Unkown file format\n"); exit(1); }
	int m = mg.num_vertices();
	IndexT* row_offsets = mg.out_rowptr();
	IndexT** index = new IndexT*[m+1];
	for (IndexT n = 0; n < m + 1; n ++) index[n] = mg.out_colidx() + row_offsets[n];
	//graph.Setup(mg.num_vertices(), mg.num_edges(), mg.out_rowptr(), mg.out_colidx());
	ValueT * vlabels = NULL;
	if (enable_label) {
		vlabels = (ValueT *)malloc(m * sizeof(ValueT));
		for (IndexT i = 0; i < m; i ++)
			vlabels[i] = mg.get_label(i);
	}
	graph.Setup(m, mg.num_edges(), mg.out_rowptr(), index, mg.out_colidx(), vlabels);
	/*
	for (IndexT i = 0; i < m; i ++) {
		IndexT row_begin = graph.edge_begin(i);
		IndexT row_end = graph.edge_end(i);
		std::cout << "vertex " << i << ": label = " << " " << " edgelist = [ ";
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = graph.getEdgeDst(offset);
			std::cout << dst << " ";
		}
		std::cout << "]" << std::endl;
	}
	*/
	return mg.nlabels();
}

