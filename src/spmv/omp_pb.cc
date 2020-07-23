// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include <vector>
//#include "simd_utils.h"
#include "prop_blocking.h"
#define SPMV_VARIANT "omp_pb" // propagation blocking

// m: number of vertices, nnz: number of non-zero values
void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *Ap, IndexT *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degrees) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);
	int num_bins = (m-1) / BIN_WIDTH + 1; // the number of bins is the number of vertices in the graph divided by the bin width
	preprocessing(m, nnz, ApT, AjT);
	double total_time = 0;

#ifdef ALIGNED
	vector<vector<aligned_vector<ValueT> > > local_value_bins(num_threads);
	vector<vector<aligned_vector<ValueT> > > value_bufs(num_threads);
	vector<vector<size_t> > buf_count(num_threads);
	for (int tid = 0; tid < num_threads; tid ++) {
		local_value_bins[tid].resize(num_bins);
		value_bufs[tid].resize(num_bins);
		buf_count[tid].resize(num_bins);
		for (int bid = 0; bid < num_bins; bid ++) {
			value_bufs[tid][bid].resize(buf_size);
			buf_count[tid][bid] = 0;
		}
	}
#endif

	Timer t;
	t.Start();
	#pragma omp parallel for schedule(dynamic, 64)
	for (int u = 0; u < m; u ++) {
		//int tid = omp_get_thread_num();
		IndexT row_begin = ApT[u];
		IndexT row_end = ApT[u+1];
		ScoreT c = x[u];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT v = AjT[offset];
			ValueT value = AxT[offset];
			int dest_bin = v >> BITS; // v / BIN_WIDTH
			value_bins[dest_bin][pos[offset]] = c * value;
			//addr[offset][0] = c * value;
		}
	}
	t.Stop();
	total_time += t.Millisecs();
	printf("\truntime [binning] = %f ms.\n", t.Millisecs());

	t.Start();
	#pragma omp parallel for schedule(dynamic, 32)
	for (int bid = 0; bid < num_bins; bid ++) {
		for(int k = 0; k < sizes[bid]; k++) {
			ScoreT c = value_bins[bid][k];
			IndexT v = vertex_bins[bid][k];
			y[v] = y[v] + c;
		}
	}
	t.Stop();
	printf("\truntime [accumulate] = %f ms.\n", t.Millisecs());
	total_time += t.Millisecs();
	printf("\truntime [total] = %f ms.\n", total_time);
	return;
}

