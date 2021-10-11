// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"
#include <vector>
#define SGD_VARIANT "omp_base"

inline ScoreT rmse_par(int m, int nnz, ScoreT *errors) {
	ScoreT total_error = 0.0;
	#pragma omp parallel for reduction(+ : total_error)
	for(int i = 0; i < m; i ++) {
		total_error += errors[i];
	}
	total_error = sqrt(total_error/nnz);
	return total_error;
}

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, int * ordering) {
	int num_threads = 1;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SGD solver (%d threads) ...\n", num_threads);
	vector<ScoreT> user_error(m*K, 0);
	vector<ScoreT> item_error(n*K, 0);
#ifdef COMPUTE_ERROR
	ScoreT *squared_errors = (ScoreT *)malloc(m * sizeof(ScoreT));
	ScoreT total_error = 0.0;
#endif

	int iter = 0;
	Timer t;
	t.Start();
	do {
		iter ++;
#ifdef COMPUTE_ERROR
		for (int i = 0; i < m; i ++) squared_errors[i] = 0;
#endif
		#pragma omp parallel for schedule(dynamic, 64)
		for(int dst = 0; dst < m; dst ++) {
			int row_begin = row_offsets[dst];
			int row_end = row_offsets[dst+1];
			int user_offset = K*dst;
			LatentT *ulv = &user_lv[user_offset];
			//LatentT *uerr = &user_error[user_offset];
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int src = column_indices[offset];
				int item_offset = K*src;
				LatentT *ilv = &item_lv[item_offset];
				//LatentT *ierr = &item_error[item_offset];
				ScoreT estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += ulv[i] * ilv[i];
				}
				ScoreT delta = rating[offset] - estimate;
#ifdef COMPUTE_ERROR
				squared_errors[dst] += delta * delta;
#endif
				for (int i = 0; i < K; i++) {
					LatentT p_d = ulv[i];
					LatentT p_s = ilv[i];
					ulv[i] += step * (-lambda * p_d + p_s * delta);
					ilv[i] += step * (-lambda * p_s + p_d * delta);
					//uerr[i] += ilv[i] * delta;
					//ierr[i] += ulv[i] * delta;
				}
			}
		}
/*
		#pragma omp parallel for
		for(int u = 0; u < m; u ++) {
			int user_offset = K*u;
			LatentT *ulv = &user_lv[user_offset];
			LatentT *uerr = &user_error[user_offset];
			for (int i = 0; i < K; i ++) {
				ulv[i] += step * (-lambda * ulv[i] + uerr[i]);
				uerr[i] = 0;
			}
		}
		#pragma omp parallel for
		for(int u = 0; u < n; u ++) {
			int item_offset = K*u;
			LatentT *ilv = &item_lv[item_offset];
			LatentT *ierr = &item_error[item_offset];
			for (int i = 0; i < K; i ++) {
				ilv[i] += step * (-lambda * ilv[i] + ierr[i]);
				ierr[i] = 0;
			}
		}
//*/
#ifdef COMPUTE_ERROR
		total_error = rmse_par(m, nnz, squared_errors);
		printf("Iteration %d: RMSE error = %f\n", iter, total_error);
		if (total_error < epsilon) break;
#endif
	} while (iter < max_iters);
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
#ifdef COMPUTE_ERROR
	free(squared_errors);
#endif
	return;
}
