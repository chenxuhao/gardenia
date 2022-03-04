// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include <omp.h>
#include "motif.h"
#include "timer.h"
#define USE_PID
#define USE_WEDGE
#define USE_SIMPLE
#define VERTEX_INDUCED
#include "mining/vertex_miner.h"
#define MOTIF_VARIANT "omp_base"

void MotifSolver(Graph &g, unsigned k, std::vector<AccType> &acc) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("cxh debug...\n");
	VertexMiner miner(&g, k, num_threads);
	EmbeddingList emb_list;
	emb_list.init(g, k);
	size_t npatterns = acc.size();
	std::vector<UlongAccu> accumulators(npatterns);
	for (size_t i = 0; i < npatterns; i++) accumulators[i].resize(num_threads);
	printf("Launching OpenMP Motif solver (%d threads) ...\n", num_threads);

	Timer t;
	t.Start();
	unsigned level = 1;
	while (level < k-2) {
		emb_list.printout_embeddings(level);
		miner.extend_vertex(level, emb_list);
		level ++;
	}
	if (k < 5) {
		emb_list.printout_embeddings(level);
		miner.aggregate(level, emb_list, accumulators);
		miner.printout_motifs(accumulators);
	} else {
		printf("Not supported\n");
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", MOTIF_VARIANT, t.Millisecs());
}
