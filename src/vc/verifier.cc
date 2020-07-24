// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "vc.h"
#include "timer.h"
#include <vector>
#include <stdlib.h>

int first_fit(Graph &g, int *colors) {
  auto m = g.V();
	int max_color = 0;
	std::vector<int> mark(m, -1);
	for (int u = 0; u < m; u++) {
    for (auto v : g.N(u))
			mark[colors[v]] = u;
		int vertex_color = 0;
		while(vertex_color < max_color && mark[vertex_color] == u)
			vertex_color++;
		if(vertex_color == max_color)
			max_color++;
		colors[u] = vertex_color;
	}
	return max_color;
}

void VCVerifier(Graph &g, int *colors_test) {
  auto m = g.V();
	printf("Verifying...\n");
	bool correct = true;
	int *colors = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
	Timer t;
	t.Start();
	int num_colors = first_fit(g, colors);
	t.Stop();
	for (int src = 0; src < m; src ++) {
    for (auto dst : g.N(src)) {
			if (dst != src && colors_test[src] == colors_test[dst]) {
				correct = false;
				break;
			}
		}
	}
	printf("\truntime [serial] = %f ms, num_colors = %d.\n", t.Millisecs(), num_colors);
	if (correct) printf("Correct\n");
	else printf("Wrong\n");
}

