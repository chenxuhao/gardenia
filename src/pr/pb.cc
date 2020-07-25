// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#include <vector>
#include "prop_blocking.h"
#define PR_VARIANT "pb" // propagation blocking

// m: number of vertices, nnz: number of non-zero values
void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching PR solver (%d threads) using propogation blocking...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	//ScoreT *sums = (ScoreT *) malloc(m * sizeof(ScoreT));
	//for (int i = 0; i < m; i ++) { sums[i] = 0; }
	vector<ScoreT> sums(m, 0);
	int num_bins = (m-1) / BIN_WIDTH + 1; // the number of bins is the number of vertices in the graph divided by the bin width

	int iter = 0;
	double error = 0;
	preprocessing(m, nnz, out_row_offsets, out_column_indices);

	Timer t;
	t.Start();
	// the following iterations
	do {
		iter ++;
		Timer tt;
		tt.Start();
		#pragma omp parallel for schedule(dynamic, 1024)
		for (int u = 0; u < m; u ++) {
			const IndexT row_begin = out_row_offsets[u];
			const IndexT row_end = out_row_offsets[u+1];
			//int degree = row_end - row_begin;
			int degree = degrees[u];
			ScoreT c = scores[u] / (ScoreT)degree; // contribution
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT v = out_column_indices[offset];
				int dest_bin = v >> BITS; // v / BIN_WIDTH
				value_bins[dest_bin][pos[offset]] = c;
			}
		}
		tt.Stop();
		if (iter == 1) printf("\truntime [binning] = %f ms.\n", tt.Millisecs());
		tt.Start();
/*
		#pragma omp parallel
		for (int bid = 0; bid < num_bins; bid ++) {
			vector<ScoreT> temp(BIN_WIDTH, 0);
			for (int k = 0; k < sizes[bid]; k++) {
				ScoreT c = value_bins[bid][k];
				IndexT v = vertex_bins[bid][k];
				int id = v % BIN_WIDTH;
				temp[id] += c;
			}
			int start = bid << BITS; // bid * BIN_WIDTH;
			for (int id = 0; id < BIN_WIDTH; id ++) {
				if(start+id < m)
					sums[start+id] = temp[id];
			}
		}
//*/
///*
		int num = (num_bins - 1) / num_threads + 1;
		#pragma omp parallel
		{
		int tid = omp_get_thread_num();
		int start = tid * num;
		int end = (tid+1) * num;
		for (int bid = start; bid < end; bid ++) {
			if(bid < num_bins) {
				for(int k = 0; k < sizes[bid]; k++) {
					ScoreT c = value_bins[bid][k];
					IndexT v = vertex_bins[bid][k];
					sums[v] = sums[v] + c;
				}
			}
		}
		}
//*/
/*
		for (int bid = 0; bid < num_bins; bid ++) {
			for(int k = 0; k < sizes[bid]; k++) {
				ScoreT c = value_bins[bid][k];
				IndexT v = vertex_bins[bid][k];
				sums[v] = sums[v] + c;
			}
		}
//*/
		tt.Stop();
		if (iter == 1) printf("\truntime [accumulate] = %f ms.\n", tt.Millisecs());
		tt.Start();
		error = 0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++) {
			ScoreT new_score = base_score + kDamp * sums[u];
			error += fabs(new_score - scores[u]);
			scores[u] = new_score;
			sums[u] = 0;
		}
		tt.Stop();
		if (iter == 1) printf("\truntime [l1norm] = %f ms.\n", tt.Millisecs());
		printf(" %2d    %lf\n", iter, error);
		if (error < EPSILON) break;
	} while(iter < MAX_ITER);
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}

