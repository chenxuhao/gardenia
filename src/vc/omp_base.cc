// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "vc.h"
#include <omp.h>
#include "timer.h"
#include "worklist.h"

void first_fit(Graph &g, Worklist &inwl, int *colors) {
  auto m = g.V();
	auto start = inwl.start;
	auto end = inwl.end;
	#pragma omp parallel for
	for (auto i = start; i < end; i++) {
		auto u = inwl.getItem(i);
		int forbiddenColors[MAXCOLOR];
		for (int i = 0; i < MAXCOLOR; i++)
      forbiddenColors[i] = m + 1;
    for (auto v : g.N(u))
			forbiddenColors[colors[v]] = u;
		int vertex_color = 0;
		while (vertex_color < MAXCOLOR && 
           forbiddenColors[vertex_color] == u)
			vertex_color++;
		assert(vertex_color < MAXCOLOR);
		colors[u] = vertex_color;
	}
}

void conflict_resolve(Graph &g, Worklist &inwl, Worklist &outwl, int *colors) {
	auto start = inwl.start;
	auto end = inwl.end;
	#pragma omp parallel for
	for (auto id = start; id < end; id ++) {
		auto src = inwl.getItem(id);
    for (auto dst : g.N(src)) {
			if (src < dst && colors[src] == colors[dst]) {
				outwl.push(src);
				break;
			}
		}
	}
}

int VCSolver(Graph &g, int *colors) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP VC solver (%d threads) ...\n", num_threads);
	Worklist inwl, outwl, *inwlptr, *outwlptr, *tmp;
	int iter = 0;
  auto m = g.V();
	inwl.ensureSpace(m);
	outwl.ensureSpace(m);
	inwlptr = &inwl;
	outwlptr = &outwl;
	int *range = (int *)malloc(m * sizeof(int));
	for (int j = 0; j < m; j++) range[j] = j;
	Timer t;
	t.Start();
	inwl.pushRange((unsigned *)range, (VertexId)m);
	int wlsz = inwl.getSize();
	while (wlsz) {
		++ iter;
		//printf("iteration=%d, wlsz=%d\n", iteration, wlsz);
		first_fit(g, *inwlptr, colors);
		conflict_resolve(g, *inwlptr, *outwlptr, colors);
		wlsz = outwlptr->getSize();
		tmp = inwlptr; inwlptr = outwlptr; outwlptr = tmp;
		outwlptr->clear();
	}
	t.Stop();
	int max_color = 0;
	#pragma omp parallel for reduction(max : max_color)
	for (int n = 0; n < m; n ++)
		max_color = max(max_color, colors[n]);
	int num_colors = max_color+1;
	//printf("\titerations = %d.\n", iter);
	printf("\truntime [omp_base] = %f ms, num_colors = %d.\n", t.Millisecs(), num_colors);
	return num_colors;
}

