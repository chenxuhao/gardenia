// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"
#include "omp_target_config.h"
#define SGD_VARIANT "omp_target"

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *h_user_lv, LatentT *h_item_lv, int *ordering) {
#ifdef USE_ICC
	LatentT *user_lv = (LatentT *) _mm_malloc(m*K*sizeof(LatentT), 64);
	LatentT *item_lv = (LatentT *) _mm_malloc(n*K*sizeof(LatentT), 64);
#else
	LatentT *user_lv = (LatentT *) aligned_alloc(64, m*K*sizeof(LatentT));
	LatentT *item_lv = (LatentT *) aligned_alloc(64, n*K*sizeof(LatentT));
#endif
	for (int i = 0; i < m * K; i++) user_lv[i] = h_user_lv[i];
	for (int i = 0; i < n * K; i++) item_lv[i] = h_item_lv[i];
	warm_up();
	Timer t;
	t.Start();
	double t1 = 0, t2 = 0;
#pragma omp target data device(0) map(tofrom:user_lv[0:m*K]) map(tofrom:item_lv[0:n*K]) map(to:row_offsets[0:(m+1)]) map(to:ordering[0:m]) map(to:column_indices[0:nnz]) map(to:rating[0:nnz]) map(to:m,lambda,step)
{
	#pragma omp target device(0)
	{
	int iter = 0;
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
		#pragma omp parallel for schedule(dynamic, 64)
		for(int i = 0; i < m; i ++) {
			//int src = ordering[i];
			int src = i;
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
#ifdef COMPUTE_ERROR
				squared_errors[src] += delta * delta;
#endif
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
#ifdef COMPUTE_ERROR
		total_error = 0.0;
		#pragma omp parallel for reduction(+ : total_error)
		for(int i = 0; i < m; i ++) {
			total_error += squared_errors[i];
		}
		total_error = sqrt(total_error/nnz);
		//printf("Iteration %d: RMSE error = %f\n", iter, total_error);
		if (total_error < epsilon) break;
#endif
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
