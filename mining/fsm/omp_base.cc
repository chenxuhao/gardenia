// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include <omp.h>
#include "fsm.h"
#include "timer.h"
#include "miner.h"

void init(int m, int num_threads, IndexT *row_offsets, IndexT *column_indices, int *labels, EmbeddingQueue &queue) {
	printf("\n=============================== Init ===============================\n\n");
	std::vector<EmbeddingQueue> lqueue(num_threads);
	#pragma omp parallel for
	for (int src = 0; src < m; src ++) {
		int tid = omp_get_thread_num();
		int src_label = labels[src];
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src+1];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			int dst_label = labels[dst];
			std::vector<Element_In_Tuple> out_update_tuple;
			out_update_tuple.push_back(Element_In_Tuple(src, 0, src_label));
			out_update_tuple.push_back(Element_In_Tuple(dst, 0, dst_label));
			if(!Pattern::is_automorphism_init(out_update_tuple)) {
				lqueue[tid].push_back(out_update_tuple);
			}
		}
	}
	for (int i = 0; i < num_threads; i ++) {
		for (int j = 0; j < lqueue[i].size(); j ++) {
			queue.push_back(lqueue[i][j]);
		}
	}
}

void construct_edgelist(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, int *labels, std::vector<LabeledEdge> &edgelist) {
	for (int src = 0; src < m; src ++) {
		int src_label = labels[src];
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src+1];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			int dst_label = labels[dst];
			LabeledEdge e(src, dst, src_label, dst_label);
			edgelist.push_back(e);
		}
	}
	assert(edgelist.size() == nnz);
}

void FSMSolver(int m, int nnz, int minsup, int k, IndexT *row_offsets, IndexT *column_indices, int *labels, long long *total) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP FSM solver (%d threads) ...\n", num_threads);
	std::vector<LabeledEdge> edge_list;
	construct_edgelist(m, nnz, row_offsets, column_indices, labels, edge_list);
	int sizeof_tuple = 2 * sizeof(Element_In_Tuple);
	EmbeddingQueue queue, filtered_queue;
	Miner miner(true, sizeof_tuple, m, nnz, edge_list, num_threads);
	miner.set_threshold(minsup);
	int level = 1;

	Timer t;
	t.Start();
	init(m, num_threads, row_offsets, column_indices, labels, queue);
	if(DEBUG) miner.printout_queue(0, queue);
	std::cout << "\n----------------------------------- aggregating -----------------------------------\n";
	miner.aggregate_parallel(queue);
	std::cout << "\n------------------------------------ filtering ------------------------------------\n";
	miner.filter(queue, filtered_queue);
	if(DEBUG) miner.printout_queue(0, filtered_queue);

	while (!filtered_queue.empty() && level < k) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- joining -------------------------------------\n";
		std::vector<EmbeddingQueue> lqueue(num_threads);
		#pragma omp parallel for
		for (int i = 0; i < filtered_queue.size(); i ++) {
			int tid = omp_get_thread_num();
			miner.join_each(k, filtered_queue[i], lqueue[tid]);
		}
		queue.clear();
		for (int i = 0; i < num_threads; i ++) {
			for (int j = 0; j < lqueue[i].size(); j ++) {
				queue.push_back(lqueue[i][j]);
			}
		}
		if(DEBUG) miner.printout_queue(level, queue);
		miner.update_tuple_size();
		std::cout << "\n----------------------------------- aggregating -----------------------------------\n";
		miner.aggregate_parallel(queue);
		//miner.aggregate(queue);
		if(DEBUG) miner.printout_agg();
		std::cout << "\n------------------------------------ filtering ------------------------------------\n";
		filtered_queue.clear();
		//miner.filter(queue, filtered_queue);
		miner.filter_parallel(queue, filtered_queue);
		if(DEBUG) miner.printout_queue(level, filtered_queue);
		level ++;
	}

	t.Stop();
	printf("\n\truntime [%s] = %f ms.\n", "omp_base", t.Millisecs());
	printf("Number of frequent subgraphs (minsup=%d): %lld\n", minsup, miner.get_frequent_patterns_count());
}
