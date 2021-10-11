// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

#include "kcl.h"
#include <omp.h>
#include "timer.h"
#include "subgraph.h"
#define KCL_VARIANT "omp_base"

void mksub(unsigned n, unsigned core, unsigned *cd, unsigned *adj, unsigned u, subgraph* &sg, unsigned char k) {
	static unsigned *old = NULL, *mynew = NULL;//to improve
	#pragma omp threadprivate(mynew,old)
	if (old == NULL) {
		mynew = (unsigned *)malloc(n * sizeof(unsigned));
		old = (unsigned *)malloc(core * sizeof(unsigned));
		for (unsigned i = 0; i < n; i ++) mynew[i] = (unsigned)-1;
	}
	for (unsigned i = 0; i < sg->n[k-1]; i ++) sg->lab[i] = 0;
	unsigned v;
	unsigned j = 0;
	for (unsigned i = cd[u]; i < cd[u+1]; i ++) {
		v = adj[i];
		mynew[v] = j;
		old[j] = v;
		sg->lab[j] = k-1;
		sg->vertices[k-1][j] = j;
		sg->d[k-1][j] = 0;//new degrees
		j ++;
	}
	sg->n[k-1] = j;
	for (unsigned i = 0; i < sg->n[k-1]; i ++) {//reodering adjacency list and computing new degrees
		v = old[i];
		for (unsigned l = cd[v]; l < cd[v+1]; l ++) {
			unsigned w = adj[l];
			j = mynew[w];
			if (j != (unsigned)-1) {
				sg->adj[sg->core*i+sg->d[k-1][i]++] = j;
			}
		}
	}
	for (unsigned i = cd[u]; i < cd[u+1]; i ++)
		mynew[adj[i]] = (unsigned)-1;
}

void kclique_thread(unsigned l, subgraph * &sg, long long *n) {
	if (l == 2) {
		for(unsigned i = 0; i < sg->n[2]; i++) { //list all edges
			unsigned u = sg->vertices[2][i];
			unsigned end = u * sg->core + sg->d[2][u];
			for (unsigned j = u * sg->core; j < end; j ++) {
				(*n) ++; //listing here!!!
			}
		}
		return;
	}
	printf("TODO\n");
	for(unsigned i = 0; i < sg->n[l]; i ++) {
		unsigned u = sg->vertices[l][i];
		//printf("%u %u\n",i,u);
		sg->n[l-1] = 0;
		unsigned end = u*sg->core+sg->d[l][u];
		for (unsigned j = u*sg->core; j < end; j ++) {//relabeling vertices and forming U'.
			unsigned v = sg->adj[j];
			if (sg->lab[v] == l) {
				sg->lab[v] = l-1;
				sg->vertices[l-1][sg->n[l-1]++] = v;
				sg->d[l-1][v] = 0;//new degrees
			}
		}
		for (unsigned j = 0; j < sg->n[l-1]; j ++) {//reodering adjacency list and computing new degrees
			unsigned v = sg->vertices[l-1][j];
			end = sg->core * v + sg->d[l][v];
			for (unsigned k = sg->core * v; k < end; k ++) {
				unsigned w = sg->adj[k];
				if (sg->lab[w] == l-1) {
					sg->d[l-1][v] ++;
				}
				else {
					sg->adj[k--] = sg->adj[--end];
					sg->adj[end] = w;
				}
			}
		}
		kclique_thread(l-1, sg, n);
		for (unsigned j = 0; j < sg->n[l-1]; j ++) {//restoring labels
			unsigned v = sg->vertices[l-1][j];
			sg->lab[v] = l;
		}
	}
}

void KCLSolver(Graph &g, unsigned k, long long *total) {
	//printf("core value (max truncated degree) = %d\n", core);
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP KCL solver (%d threads) ...\n", num_threads);
	//subgraph *sg;
	long long num = 0;

	Timer t;
	t.Start();
	//#pragma omp parallel private(sg) reduction(+:num)
	#pragma omp parallel reduction(+:num)
	{
		//subgraph *sg = (subgraph *)malloc(sizeof(subgraph));
		subgraph *sg = new subgraph;
		sg->allocate(g.core, k);
		#pragma omp for schedule(dynamic, 1) nowait
		for (unsigned i = 0; i < g.n; i ++) {
			mksub(g.n, g.core, g.cd, g.adj, i, sg, k);
			kclique_thread(k-1, sg, &num);
		}
	}
	t.Stop();

	*total = num;
	printf("Number of %d-cliques: %lld\n", k, num);
	printf("\truntime [%s] = %f ms.\n", KCL_VARIANT, t.Millisecs());
	return;
}
