// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "vc.h"
#include "timer.h"
#include <vector>
#include <stdlib.h>

int first_fit(int m, int *row_offsets, int *column_indices, int *colors) {
	int max_color = 0;
	std::vector<int> mark(m, -1);
	for(int vertex = 0; vertex < m; vertex++) {
		int row_begin = row_offsets[vertex];
		int row_end   = row_offsets[vertex + 1];
		for(int offset = row_begin; offset < row_end; offset++) {
			int neighbor = column_indices[offset];
			mark[colors[neighbor]] = vertex;
		}
		int vertex_color = 0;
		while(vertex_color < max_color && mark[vertex_color] == vertex)
			vertex_color++;
		if(vertex_color == max_color)
			max_color++;
		colors[vertex] = vertex_color;
	}
	return max_color;
}

void VCVerifier(int m, int *row_offsets, int *column_indices, int *colors_test) {
	printf("Verifying...\n");
	bool correct = true;
	int *colors = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i ++) colors[i] = MAXCOLOR;
	Timer t;
	t.Start();
	int num_colors = first_fit(m, row_offsets, column_indices, colors);
	t.Stop();
	for (int src = 0; src < m; src ++) {
		for (int offset = row_offsets[src]; offset < row_offsets[src + 1]; offset ++) {
			int dst = column_indices[offset];
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

void write_solution(int m, char *fname, int *colors) {
	FILE *fp = fopen(fname, "w");
	for (int i = 0; i < m; i++) {
		fprintf(fp, "%d\n", colors[i]);
	}
	fclose(fp);
}

