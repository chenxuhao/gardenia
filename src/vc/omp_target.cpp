// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "vc.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define VC_VARIANT "omp_target"

#pragma omp declare target
#include "worklist.h"

void first_fit(int m, int *row_offsets, int *column_indices, Worklist &inwl, int *colors) {
	int start = inwl.start;
	int end = inwl.end;
	#pragma omp parallel for
	for (int i = start; i < end; i++) {
		int vertex = inwl.getItem(i);
		int forbiddenColors[MAXCOLOR];
		for(int i = 0; i < MAXCOLOR; i++) forbiddenColors[i] = m + 1;
		int row_begin = row_offsets[vertex];
		int row_end = row_offsets[vertex + 1];
		for (int offset = row_begin; offset < row_end; offset++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[color] = vertex;
		}
		int vertex_color = 0;
		while (vertex_color < MAXCOLOR && forbiddenColors[vertex_color] == vertex)
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
#pragma omp end declare target

int VCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *colors) {
	int iter = 0;
	Timer t;
	t.Start();
	warm_up();
	double t1, t2;
	
#pragma omp target data device(0) map(tofrom:colors[0:m]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz]) map(to:m,nnz)
{
	#pragma omp target device(0)
	{
	Worklist inwl, outwl, *inwlptr, *outwlptr, *tmp;
	inwl.ensureSpace(m);
	outwl.ensureSpace(m);
	inwlptr = &inwl;
	outwlptr = &outwl;
	int *range = (int *)malloc(m * sizeof(int));
	for (int j = 0; j < m; j++) range[j] = j;
	inwl.pushRange((unsigned *)range, (unsigned)m);
	int wlsz = inwl.getSize();
	t1 = omp_get_wtime();
	while (wlsz) {
		++ iter;
		//printf("iteration=%d, wlsz=%d\n", iteration, wlsz);
		first_fit(m, row_offsets, column_indices, *inwlptr, colors);
		conflict_resolve(m, row_offsets, column_indices, *inwlptr, *outwlptr, colors);
		wlsz = outwlptr->getSize();
		tmp = inwlptr; inwlptr = outwlptr; outwlptr = tmp;
		outwlptr->clear();
	}
	t2 = omp_get_wtime();
	}
}
	t.Stop();
	int max_color = 0;
	#pragma omp parallel for reduction(max : max_color)
	for (int n = 0; n < m; n ++)
		max_color = max(max_color, colors[n]);
	int num_colors = max_color+1;
	printf("\titerations = %d.\n", iter);
	//printf("\truntime [%s] = %f ms, num_colors = %d.\n", VC_VARIANT, t.Millisecs(), num_colors);
	printf("\truntime [%s] = %f ms, num_colors = %d.\n", VC_VARIANT, 1000*(t2-t1), num_colors);
	return num_colors;
}

