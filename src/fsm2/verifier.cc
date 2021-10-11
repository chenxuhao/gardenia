#include "fsm.h"
#include "timer.h"
#include "verifier.h"

void FSMVerifier(const Graph &graph, int minsup, size_t test_total) {
	printf("Verifying...\n");
	Verifier miner(graph);
	std::vector<Edge> edges;
	Embeddings_map3 root;
	Timer t;
	t.Start();
	for (unsigned int from = 0; from < graph.size(); ++from) {
		if (get_forward_root(graph, graph[from], edges)) {   // get the edge list of the node g[from] in graph g
			for (std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it) {
				root[graph[from].label][it->elabel][graph[it->to].label].push(*it, 0);  // insert current edge and null prev Emb
			}
		} // if
	} // for from
	miner.grow(root, minsup);
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());
	size_t count = miner.get_count();
	if (count == test_total) printf("Correct\n");
	else printf("Number of frequent subgraphs (minsup=%d): %ld, but test_total = %ld\n", minsup, count, test_total);
}

