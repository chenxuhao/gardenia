// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"
#include "omp_target_config.h"
#define SGD_VARIANT "omp_target_fusion"

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, int *ordering) {
	int iter = 0;
	warm_up();

	Timer t;
	t.Start();
	double t1, t2;
	ScoreT *h_error = (ScoreT *) malloc(sizeof(ScoreT));
	h_error[0] = 0;
#pragma omp target data device(0) map(tofrom:user_lv[0:m*K]) map(tofrom:item_lv[0:n*K]) map(to:row_offsets[0:(m+1)]) map(to:ordering[0:m]) map(to:column_indices[0:nnz]) map(to:rating[0:nnz]) map(to:m,lambda,step)
{
#ifdef COMPUTE_ERROR
	ScoreT* squared_errors = (ScoreT *)malloc(m * sizeof(ScoreT));
	ScoreT total_error = 0.0;
#endif
	t1 = omp_get_wtime();
	do {
		iter ++;
#ifdef COMPUTE_ERROR
		for (int i = 0; i < m; i ++) squared_errors[i] = 0;
#endif
		#pragma omp target device(0)
		#pragma omp parallel for schedule(dynamic, 64)
		for(int i = 0; i < m; i ++) {
			//int src = ordering[i];
			int src = i;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src+1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				ScoreT estimate = 0;
				#pragma ivdep
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				ScoreT delta = rating[offset] - estimate;
#ifdef COMPUTE_ERROR
				squared_errors[src] += delta * delta;
#endif
				#pragma ivdep
				for (int i = 0; i < K; i++) {
					LatentT p_s = user_lv[src*K+i];
					LatentT p_d = item_lv[dst*K+i];
					user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
					item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
				}
			}
		}
#ifdef COMPUTE_ERROR
		#pragma omp target device(0) map(from:h_error[0:1])
		//#pragma omp target device(0)
		{
		total_error = 0.0;
		#pragma omp parallel for reduction(+ : total_error)
		for(int i = 0; i < m; i ++)
			total_error += squared_errors[i];
		total_error = sqrt(total_error/nnz);
		h_error[0] = total_error;
		}
		error = h_error[0];
		error = sqrt(error/nnz);
		printf("Iteration %d: RMSE error = %f per edge\n", iter, error);
		if (error < epsilon) break;
#endif
	} while (iter < max_iters);
	t2 = omp_get_wtime();
}
	t.Stop();
	printf("\titerations = %d.\n", iter);
	//printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, 1000*(t2-t1));
	return;
}
