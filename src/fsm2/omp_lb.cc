// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "fsm.h"
#include <omp.h>
#include <timer.h>
#include <lb.hpp>
#include "miner_omp_lb.h"
#define FSM_VARIANT "omp_lb"

void FSMSolver(const Graph &graph, int minsup, unsigned k, size_t &total) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP-Base FSM solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();
	Miner_omp miner(graph, num_threads, minsup, k);
	EdgeList edges;
	Projected_map3 root;
	int m = 0, nnz = 0;
	int single_edge_dfscodes = 0;
	bool computation_end = false;
	for(unsigned int from = 0; from < graph.size(); ++from) {
		if(get_forward_root(graph, graph[from], edges)) {   // get the edge list of the node g[from] in graph g
			for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
				if(root.count(graph[from].label) == 0 || root[graph[from].label].count((*it)->elabel) == 0 || root[graph[from].label][(*it)->elabel].count(graph[(*it)->to].label) == 0) {
					single_edge_dfscodes++;
				}
				root[graph[from].label][(*it)->elabel][graph[(*it)->to].label].push(0, *it, 0);
				nnz ++;
			} //for
		} // if
		m ++;
	} // for from
	std::cout << "Total number of nodes " << m << std::endl;
	std::cout << "Total number of edges " << nnz << std::endl;
	std::cout << "Single edge DFScodes " << single_edge_dfscodes << std::endl;
    int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / num_threads);
	//miner.grow(root);
	Thread_private_data gprv;
	#pragma omp parallel num_threads(num_threads) private(gprv)
	{
		int thread_id = omp_get_thread_num();
		gprv.thread_id = thread_id;
		gprv.current_dfs_level = 0;
		gprv.task_split_level = 0;
		gprv.embeddings_regeneration_level = 0;
		gprv.is_running = true;

		Projected emb_prv;
		if (thread_id == 0) std::cout << "dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
		int start_index = thread_id * dfscodes_per_thread;
		int end_index = start_index + dfscodes_per_thread - 1;
		if (end_index > single_edge_dfscodes - 1)
			end_index = single_edge_dfscodes - 1;
		if(start_index <= end_index) 
			miner.activate_thread(thread_id);
		std::deque<DFS> tmp;
		gprv.dfs_task_queue.push_back(tmp);
		int index = 0;
		for(Projected_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
			for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
				for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					if( index >= start_index && index <= end_index ) {
						// Build the initial two-node graph.  It will be grown recursively within project.
						DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
						gprv.dfs_task_queue[0].push_back(dfs);
					}
					index++;
				} // for tolabel
			} // for elabel
		} // for fromlabel
#if 0
		//#pragma omp critical
		//std::cout << "thread id " << thread_id << " starting" << std::endl;
		//#pragma omp barrier
		//std::deque<DFS> queue = gprv.dfs_task_queue[0];
		//for (auto it = queue.begin(); it < queue.end(); it++) {
			//DFS dfs = *it;
			//queue.erase(it);
		while (gprv.dfs_task_queue[0].size() != 0) {
			DFS dfs = gprv.dfs_task_queue[0].front();
			gprv.dfs_task_queue[0].pop_front();
			gprv.DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			miner.project(root[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, gprv);
			gprv.DFS_CODE.pop();
		}
#else
		//std::cout << "thread_id = " << gprv.thread_id << std::endl;
		while (computation_end == false) {
			//if (thread_id == 0) std::cout << "size = " << gprv.dfs_task_queue[0].size() << std::endl;
			if (miner.thread_working(gprv) == false) {
				#ifdef ENABLE_LB
				if(num_threads > 1 && computation_end == false) miner.task_schedule(gprv);
				#endif
				if (thread_id == 0) {
					if (miner.all_threads_idle() == true) {
						//std::cout << "all threads are idle" << std::endl;
						//is_working = false;
						computation_end = true;
					}
				}
			} else {
				#ifdef ENABLE_LB
				if(num_threads > 1) miner.task_schedule(gprv);
				#endif
				DFS dfs = gprv.dfs_task_queue[0].front();
				gprv.dfs_task_queue[0].pop_front();
				gprv.DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
				gprv.current_dfs_level = 1;
				#ifdef ENABLE_LB
				if(gprv.embeddings_regeneration_level >= 1)
					miner.regenerate_embeddings(root[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, gprv);
				else
				#endif
					miner.project(root[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, gprv);
				gprv.current_dfs_level = 0;
				gprv.DFS_CODE.pop();
				if(gprv.dfs_task_queue[0].size() == 0) {
					miner.deactivate_thread(thread_id);
				#ifdef ENABLE_LB
					gprv.embeddings_regeneration_level = 0;
					miner.set_regen_level(thread_id, 0);
				#endif
				}
			} // end if (thread_working(gprv) == false)
		} // end while
#endif
//#pragma omp critical
//		std::cout << "thread " << thread_id << " exited" << std::endl;
//#pragma omp barrier
	}
	t.Stop();

	total = miner.get_count();
	printf("Number of frequent subgraphs (minsup=%d): %ld\n", minsup, total);
	printf("\truntime [%s] = %f ms.\n", FSM_VARIANT, t.Millisecs());
	return;
}

