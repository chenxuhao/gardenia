// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "fsm.h"
#include <timer.h>
#include <types.hpp>
#include <graph_types.hpp>
#include "miner.h"
#define FSM_VARIANT "serial"

void FSMSolver(const Graph &graph, int minsup, size_t &total) {
	printf("Launching Serial FSM solver ...\n");
	Timer t;
	t.Start();
	Miner miner(graph, minsup);
	EdgeList edges;
	Projected_map3 root;
	for(unsigned int from = 0; from < graph.size(); ++from) {
		if(get_forward_root(graph, graph[from], edges)) {   // get the edge list of the node g[from] in graph g
			for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
				//projected (PDFS vector) entry: graph id (always 0 for single graph), edge pointer and null PDFS
				root[graph[from].label][(*it)->elabel][graph[(*it)->to].label].push(0, *it, 0);
		}   // if
	}   // for from
	miner.grow(root);
	t.Stop();
	total = miner.get_count();
	printf("Number of frequent subgraphs (minsup=%d): %ld\n", minsup, total);
	printf("\truntime [%s] = %f ms.\n", FSM_VARIANT, t.Millisecs());
}
