// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include "fsm.h"
#include "timer.h"
#include "miner.h"

void init(int m, IndexT *row_offsets, IndexT *column_indices, int *labels, EmbeddingQueue &queue) {
	printf("\n============================== init ==============================\n");
	for (int src = 0; src < m; src ++) {
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
				queue.push_back(out_update_tuple);
			}
		}
	}
}

void construct_edgelist(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, int *labels, std::vector<LabeledEdge> &edgelist) {
	//printf("construct edgelist\n");
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
	std::vector<LabeledEdge> edge_list;
	construct_edgelist(m, nnz, row_offsets, column_indices, labels, edge_list);
	Timer t;
	t.Start();
	EmbeddingQueue queue, filtered_queue;
	// initialization
	init(m, row_offsets, column_indices, labels, queue);
	int sizeof_tuple = 2 * sizeof(Element_In_Tuple);
	Miner miner(true, sizeof_tuple, m, nnz, edge_list);
	miner.set_threshold(minsup);
	std::cout << "\n----------------------------------- aggregating -----------------------------------\n";
	miner.aggregate(queue);
	if(DEBUG) miner.printout_agg();
	std::cout << "\n------------------------------------ filtering ------------------------------------\n";
	miner.filter(queue, filtered_queue);
	int level = 1;
	queue.clear();

	while (!filtered_queue.empty() && level < k) {
		std::cout << "\n============================== Level " << level << " ==============================\n";
		std::cout << "\n------------------------------------- joining -------------------------------------\n";
		miner.join_all(k, filtered_queue, queue);
		//std::cout << "number of tuples after join: " << queue.size() << "\n";
		filtered_queue.clear();
		std::cout << "\n----------------------------------- aggregating -----------------------------------\n";
		miner.aggregate(queue);
		if(DEBUG) miner.printout_agg();
		std::cout << "\n------------------------------------ filtering ------------------------------------\n";
		miner.filter(queue, filtered_queue);
		queue.clear();
		level ++;
	}
	t.Stop();
	printf("\n\truntime [%s] = %f ms.\n", "serial", t.Millisecs());
	printf("Number of frequent subgraphs (minsup=%d): %lld\n", minsup, miner.get_frequent_patterns_count());
}
