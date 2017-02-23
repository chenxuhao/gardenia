// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "vc.h"
#include "timer.h"
void VCVerifier(int m, int *row_offsets, int *column_indices, int *colors) {
	bool correct = true;
	Timer t;
	t.Start();
	for (int src = 0; src < m; src ++) {
		for (int offset = row_offsets[src]; offset < row_offsets[src + 1]; offset ++) {
			int dst = column_indices[offset];
			if (colors[src] == colors[dst] && dst != src) {
				correct = false;
				break;
			}
		}
	}
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
	if (correct)
		printf("Correct\n");
	else
		printf("Wrong\n");
}

void write_solution(int m, char *fname, int *colors) {
	FILE *fp = fopen(fname, "w");
	for (int i = 0; i < m; i++) {
		fprintf(fp, "%d\n", colors[i]);
	}
	fclose(fp);
}

