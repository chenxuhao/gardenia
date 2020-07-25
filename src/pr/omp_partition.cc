// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#define PR_VARIANT "omp_partition"

#define PARTITION

#ifdef PARTITION
#include "segmenting.h"
#endif

void PRSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);

#ifdef PARTITION
	segmenting(m, row_offsets, column_indices, NULL);
#endif
	const ScoreT base_score = (1.0f - kDamp) / m;
	ScoreT *outgoing_contrib = (ScoreT *) malloc(m * sizeof(ScoreT));
	//ScoreT *sums = (ScoreT *) malloc(m * sizeof(ScoreT));
	vector<ScoreT> sums(m, 0);
	int iter;
	Timer t;
	t.Start();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		#pragma omp parallel for
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degrees[n];
#ifndef PARTITION
		#pragma omp parallel for schedule(dynamic, 64)
		for (int dst = 0; dst < m; dst ++) {
			ScoreT incoming_total = 0;
			const IndexT row_begin = row_offsets[dst];
			const IndexT row_end = row_offsets[dst+1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT src = column_indices[offset];
				incoming_total += outgoing_contrib[src];
			}
			sums[dst] = incoming_total;
		}
#else
		// parallel subgraph processing
		int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
		int num_ranges = (m - 1) / RANGE_WIDTH + 1;
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			int size = ms_of_subgraphs[bid];
			//printf("processing subgraph[%d] with %d vertices\n", bid, size);
			//printf("printing partial_sums:\n");
			#pragma omp parallel for schedule(dynamic, 64)
			for (int dst = 0; dst < size; dst ++) {
				ScoreT incoming_total = 0;
				const IndexT row_begin = rowptr_blocked[bid][dst];
				const IndexT row_end = rowptr_blocked[bid][dst+1];
				for (IndexT offset = row_begin; offset < row_end; offset ++) {
					IndexT src = colidx_blocked[bid][offset];
					incoming_total += outgoing_contrib[src];
				}
				partial_sums[bid][dst] = incoming_total;
				//printf("partial_sums[%d][%d] = %f\n", bid, dst, partial_sums[bid][dst]);
			}
		}

		// cache-aware merge
		#pragma omp parallel for schedule(dynamic, 64)
		for (int rid = 0; rid < num_ranges; rid ++) {
			for (int bid = 0; bid < num_subgraphs; bid ++) {
				int start = range_indices[bid][rid];
				int end = range_indices[bid][rid+1];
				for (int lid = start; lid < end; lid ++) {
					int gid = idx_map[bid][lid];
					ScoreT local_sum = partial_sums[bid][lid];
					sums[gid] += local_sum;
				}
			}
		}
		//for (int u = 0; u < m; u ++) printf("sums[%d] = %f\n", u, sums[u]);
#endif
		double error = 0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++) {
			ScoreT old_score = scores[u];
			scores[u] = base_score + kDamp * sums[u];
			error += fabs(scores[u] - old_score);
			sums[u] = 0;
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
