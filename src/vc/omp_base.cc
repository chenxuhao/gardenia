// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "vc.h"
#include <omp.h>
#include "timer.h"
#include "worklist.h"
#define COLOR_VARIANT "openmp"

void first_fit(int m, int *row_offsets, int *column_indices, Worklist &inwl, int *colors) {
	int start = inwl.start;
	int end = inwl.end;
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	int **forbiddenColors = (int **) malloc(num_threads*sizeof(int*));
	for (int i = 0; i < num_threads; i++) {
		forbiddenColors[i] = (int *) malloc((MAXCOLOR+1)*sizeof(int));
		for(int j = 0; j < MAXCOLOR; j++) forbiddenColors[i][j] = m + 1;
	}
	#pragma omp parallel for
	for (int i = start; i < end; i++) {
		int tid = omp_get_thread_num();
		int vertex = inwl.getItem(i);
		int row_begin = row_offsets[vertex];
		int row_end = row_offsets[vertex + 1];
		for (int offset = row_begin; offset < row_end; offset++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[tid][color] = vertex;
		}
		int vertex_color = 0;
		while (vertex_color < MAXCOLOR && forbiddenColors[tid][vertex_color] == vertex)
			vertex_color++;
		assert(vertex_color < MAXCOLOR);
		colors[vertex] = vertex_color;
	}
}

void conflict_resolve(int m, int *row_offsets, int *column_indices, Worklist &inwl, Worklist &outwl, int *colors) {
	int start = inwl.start;
	int end = inwl.end;
	#pragma omp parallel for
	for (int id = start; id < end; id ++) {
		int src = inwl.getItem(id);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = column_indices[offset];
			if (src < dst && colors[src] == colors[dst]) {
				outwl.push(src);
				break;
			}
		}
	}
}

void VCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors) {
	printf("Launching OpenMP Color solver...\n");
	//omp_set_num_threads(2);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching %d threads...\n", num_threads);
	
	Worklist inwl, outwl, *inwlptr, *outwlptr, *tmp;
	int iter = 0;
	inwl.ensureSpace(m);
	outwl.ensureSpace(m);
	inwlptr = &inwl;
	outwlptr = &outwl;
	int *range = (int *)malloc(m * sizeof(int));
	for (int j = 0; j < m; j++)
		range[j] = j;
	Timer t;
	t.Start();
	inwl.pushRange((unsigned *)range, (unsigned)m);
	int wlsz = inwl.getSize();
	while (wlsz) {
		++ iter;
		//printf("iteration=%d, wlsz=%d\n", iteration, wlsz);
		first_fit(m, row_offsets, column_indices, *inwlptr, colors);
		conflict_resolve(m, row_offsets, column_indices, *inwlptr, *outwlptr, colors);
		wlsz = outwlptr->getSize();
		tmp = inwlptr; inwlptr = outwlptr; outwlptr = tmp;
		outwlptr->clear();
	}
	t.Stop();
	int num_colors = 0;
	#pragma omp parallel for reduction(max : num_colors)
	for (int n = 0; n < m; n ++)
		num_colors = max(num_colors, colors[n]);
	printf("\titerations = %d.\n", iter);
	printf("\truntime[%s] = %f ms, num_colors = %d.\n", COLOR_VARIANT, t.Millisecs(), num_colors);
}

