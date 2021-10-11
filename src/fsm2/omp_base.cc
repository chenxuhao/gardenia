// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "fsm.h"
#include <timer.h>
#include <types.hpp>
#include <graph_types.hpp>
#include "miner_omp_base.h"
#define FSM_VARIANT "omp_base"

void FSMSolver(const Graph &graph, int minsup, unsigned k, size_t &total) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP-Queue FSM solver (%d threads) ...\n", num_threads);
	Miner_omp miner(num_threads, minsup, k);
	miner.set_graph(graph);

	Timer t;
	t.Start();
	EdgeList edges;
	Projected_map3 root;
	int single_edge_dfscodes = 0;
	for (unsigned int from = 0; from < graph.size(); ++from) {
		if (get_forward_root(graph, graph[from], edges)) {   // get the edge list of the node g[from] in graph g
			int from_label = graph[from].label;
			for (EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
				//embeddings with a single edge
				if (root.count(from_label) == 0 || root[from_label].count((*it)->elabel) == 0 || root[from_label][(*it)->elabel].count(graph[(*it)->to].label) == 0) {
					single_edge_dfscodes++;
				}
				//projected (PDFS vector) entry: graph id (always 0 for single graph), edge pointer and null PDFS
				root[graph[from].label][(*it)->elabel][graph[(*it)->to].label].push(0, *it, 0);
			} //for
		} // if
	} // for from
	std::deque<DFS> global_queue;
	for(Projected_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
		for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
			for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
				DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
				global_queue.push_back(dfs);
			} // for tolabel
		} // for elabel
	} // for fromlabel
	std::cout << "Single edge DFScodes " << single_edge_dfscodes << std::endl;
	//for (auto dfs : global_queue) std::cout << dfs << "\n";
	int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / num_threads);
	std::cout << "dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 

	Thread_private_data gprv;
	#pragma omp parallel for private(gprv) schedule(dynamic, 5)
	for (auto it = global_queue.begin(); it < global_queue.end(); it++) {
		int thread_id = omp_get_thread_num();
		gprv.thread_id = thread_id;
		gprv.current_dfs_level = 0;
		std::deque<DFS> tmp;
		gprv.dfs_task_queue.push_back(tmp);
		DFS dfs = *it;
		gprv.dfs_task_queue[0].push_back(dfs);
		gprv.DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
		miner.project(root[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, gprv);
		gprv.DFS_CODE.pop();
	}  //pragma omp
	global_queue.clear();

	t.Stop();
	total = miner.get_count();
	printf("Number of frequent subgraphs (minsup=%d): %lld\n", minsup, total);
	printf("\truntime [%s] = %f ms.\n", FSM_VARIANT, t.Millisecs());
	return;
}

