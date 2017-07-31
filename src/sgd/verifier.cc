// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"

// calculate RMSE
ScoreT compute_rmse(int m, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv) {
	ScoreT total_error = 0;
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

void SGDVerifier(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step, int *ordering, int max_iters, float epsilon) {
	printf("Verifying...\n");
	int iter = 0;
	ScoreT total_error = compute_rmse(m, nnz, row_offsets, column_indices, rating, user_lv, item_lv);
	printf("Iteration %d: RMSE error = %f per edge\n", iter, total_error);
	Timer t;
	t.Start();
	do {
		iter ++;
		for(int i = 0; i < m; i ++) {
			int src = ordering[i];
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				ScoreT estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				ScoreT delta = rating[offset] - estimate;
				for (int i = 0; i < K; i++) {
					LatentT p_s = user_lv[src*K+i];
					LatentT p_d = item_lv[dst*K+i];
					user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
					item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
				}
			}
		}
		total_error = compute_rmse(m, nnz, row_offsets, column_indices, rating, user_lv, item_lv);
		printf("Iteration %d: RMSE error = %f per edge\n", iter, total_error);
	} while (iter < max_iters && total_error > epsilon);
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
}

