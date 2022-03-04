// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include <omp.h>
#include "kcl.h"
#include "timer.h"
#define USE_SIMPLE
#define USE_BASE_TYPES
#include "mining/vertex_miner.h"

void KclSolver(Graph &g, unsigned k, uint64_t &total) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP KCL solver (%d threads) ...\n", num_threads);
	VertexMiner miner(&g, k, num_threads);
	EmbeddingList emb_list;
	emb_list.init(g, k, true);
	Accumulator<uint64_t> num(num_threads);
	Timer t;
	t.Start();
	unsigned level = 1;
	while (1) {
		emb_list.printout_embeddings(level);
		miner.extend_vertex(level, emb_list, num);
		if (level == k-2) break; 
		level ++;
	}
	total = num.reduce();
	t.Stop();
	std::cout << "\n\ttotal_num_cliques = " << total << "\n\n";
	printf("\truntime [omp_base] = %f sec\n", t.Seconds());
}

