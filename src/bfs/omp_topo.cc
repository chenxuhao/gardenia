// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include <omp.h>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include "timer.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"
#define BFS_VARIANT "omp_topo"

void bfs_step(int m, int *row_offsets, int *column_indices, vector<int> &depth, bool *visited, bool *expanded, bool &changed) {
#pragma omp parallel for
	for (int src = 0; src < m; src ++) {
		if(visited[src] && !expanded[src]) {
			expanded[src] = true;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				int curr_val = depth[dst];
				if (curr_val == MYINFINITY) { // not visited
					if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
						changed = true;
					}
				}
			}
		}
	}
#pragma omp parallel for
	for (int src = 0; src < m; src ++) {
		if(depth[src] < MYINFINITY && !visited[src]) visited[src] = true;
	}
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *in_degree, int *degree, DistT *dist) {
	//omp_set_num_threads(12);
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP BFS solver (%d threads) ...\n", num_threads);
	Timer t;
	vector<int> depth(m, MYINFINITY);
	depth[source] = 0;
	bool *visited = (bool *)malloc(m*sizeof(bool));
	bool *expanded = (bool *)malloc(m*sizeof(bool));
	memset(visited, 0, m * sizeof(bool));
	memset(expanded, 0, m * sizeof(bool));
	int iter = 0;
	visited[source] = true;
	bool changed;
	t.Start();
	do {
		++ iter;
		changed = false;
		bfs_step(m, out_row_offsets, out_column_indices, depth, visited, expanded, changed);
		//printf("iteration=%d\n", iter);
	} while(changed);
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	for(int i = 0; i < m; i ++) dist[i] = depth[i];
	return;
}
