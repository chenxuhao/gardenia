// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define SGD_VARIANT "openmp"

// calculate RMSE
inline ScoreT par_compute_rmse(int m, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv) {
	ScoreT total_error = 0;
	#pragma omp parallel for reduction(+ : total_error) schedule(dynamic, 64)
	for(int src = 0; src < m; src ++) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++) {
				estimate += user_lv[src*K+i] * item_lv[dst*K+i];
			}
			ScoreT error = rating[offset] - estimate;
			total_error += error * error;
		}
	}
	total_error = sqrt(total_error/nnz);
	return total_error;
}

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *h_user_lv, LatentT *h_item_lv, ScoreT lambda, ScoreT step, int * ordering, int max_iters, float epsilon) {
	LatentT *user_lv = (LatentT *) _mm_malloc(m*K*sizeof(LatentT), 64);
	LatentT *item_lv = (LatentT *) _mm_malloc(n*K*sizeof(LatentT), 64);
	for (int i = 0; i < m * K; i++) user_lv[i] = h_user_lv[i];
	for (int i = 0; i < n * K; i++) item_lv[i] = h_item_lv[i];
	warm_up();
	ScoreT error = par_compute_rmse(m, nnz, row_offsets, column_indices, rating, user_lv, item_lv);
	printf("Iteration 0: RMSE error = %f per edge\n", error);
	Timer t;
	t.Start();
	double t1, t2;
#pragma omp target data device(0) map(tofrom:user_lv[0:m*K]) map(tofrom:item_lv[0:n*K]) map(to:row_offsets[0:(m+1)]) map(to:ordering[0:m]) map(to:column_indices[0:nnz]) map(to:rating[0:nnz]) map(to:m,lambda,step)
{
	#pragma omp target device(0)
	{
	int iter = 0;
	ScoreT total_error = 0;
	t1 = omp_get_wtime();
	do {
		iter ++;
		#pragma omp parallel for schedule(dynamic, 64)
		for(int i = 0; i < m; i ++) {
			int src = ordering[i];
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				ScoreT estimate = 0;
				//#pragma vector aligned
				//#pragma ivdep
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				ScoreT delta = rating[offset] - estimate;
				//#pragma vector aligned
				//#pragma ivdep
				for (int i = 0; i < K; i++) {
					LatentT p_s = user_lv[src*K+i];
					LatentT p_d = item_lv[dst*K+i];
					user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
					item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
				}
			}
		}
		ScoreT delta = 0;
		#pragma omp parallel for reduction(+ : delta) schedule(dynamic, 64)
		for(int src = 0; src < m; src ++) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				ScoreT estimate = 0;
				//#pragma ivdep
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				ScoreT error = rating[offset] - estimate;
				delta += error * error;
			}
		}
		total_error = sqrt(delta/nnz);
		//printf("Iteration %d: RMSE error = %f per edge\n", iter, error);
	} while (iter < max_iters && total_error > epsilon);
	t2 = omp_get_wtime();
	}
}
	t.Stop();
	//printf("\titerations = %d.\n", iter);
	//printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, 1000*(t2-t1));
	return;
}
