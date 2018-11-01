// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"

// calculate RMSE
inline ScoreT rmse(int m, int nnz, ScoreT *errors) {
	ScoreT total_error = 0.0;
	for(int i = 0; i < m; i ++)
		total_error += errors[i];
	total_error = sqrt(total_error/nnz);
	return total_error;
}

void SGDVerifier(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, int *ordering) {
	printf("Verifying...\n");
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

		for(int i = 0; i < m; i ++) {
			//int src = ordering[i];
			int src = i;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				ScoreT estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				ScoreT delta = rating[offset] - estimate;
#ifdef COMPUTE_ERROR
				squared_errors[src] += delta * delta;
#endif
				for (int i = 0; i < K; i++) {
					LatentT p_s = user_lv[src*K+i];
					LatentT p_d = item_lv[dst*K+i];
					user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
					item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
				}
			}
		}
#ifdef COMPUTE_ERROR
		total_error = rmse(m, nnz, squared_errors);
		printf("Iteration %d: RMSE error = %f\n", iter, total_error);
		if (total_error < epsilon) break;
#endif
	} while (iter < max_iters);
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
#ifdef COMPUTE_ERROR
	free(squared_errors);
#endif
	return;
}
