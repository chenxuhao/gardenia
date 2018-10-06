// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include <stdlib.h>
#include "timer.h"
#ifdef SIM
#include "sim.h"
#include "bitmap.h"
#endif
#define PR_VARIANT "omp_base"

void PRSolver(int m, int nnz, IndexType *row_offsets, IndexType *column_indices, IndexType *out_row_offsets, IndexType *out_column_indices, int *degree, ScoreT *scores) {
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	ScoreT *outgoing_contrib = (ScoreT *) malloc(m * sizeof(ScoreT));
	int iter;
	Timer t;
	t.Start();
#ifdef SIM
	//int *hub_bitmap = (int *) malloc(m * sizeof(int));
	Bitmap hub(m);
	hub.reset();
	int num_hubs = set_hub(m, nnz, degree, hub);
	m5_checkpoint(0,0);
	set_addr_bounds(1,(uint64_t)row_offsets,(uint64_t)&row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)column_indices,(uint64_t)&column_indices[nnz],8);
	//set_addr_bounds(1,(uint64_t)scores,(uint64_t)&scores[m],8);
	set_addr_bounds(3,(uint64_t)outgoing_contrib,(uint64_t)&outgoing_contrib[m],8);
	set_addr_bounds(6,(uint64_t)hub.start_,(uint64_t)hub.end_,8);
	printf("Begin of ROI\n");
	printf("This graph has %d hub vertices\n", num_hubs);
#endif
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		#pragma omp parallel for
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
		#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			const IndexType row_begin = row_offsets[src];
			const IndexType row_end = row_offsets[src + 1];
			//#pragma omp simd reduction(+ : incoming_total)
			for (IndexType offset = row_begin; offset < row_end; offset ++) {
				IndexType dst = column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
