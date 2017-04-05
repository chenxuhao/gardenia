// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"
void shuffle(int n, int *a) {
	int index, tmp;
	srand(time(NULL));
	for(int i = 0; i < n; i ++) {
		index = rand() % (n-i) + i;
		if(index != i) {
			tmp = a[i];
			a[i] = a[index];
			a[index] = tmp;
		}
	}
}

void SGDVerifier(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv) {
	printf("Verifying...\n");
	int iter = 0;
	ScoreT total_error;
	//print_latent_vector(m, n, user_lv, item_lv);
	int *vertices = (int *)malloc(m * sizeof(int));
	for(int i = 0; i < m; i ++) vertices[i] = i;
	Timer t;
	t.Start();
	do {
		iter ++;
		// Update
		shuffle(m, vertices);
		//printf("vertices[ ");
		//for(int i = 0; i < m; i ++) printf("%d ", vertices[i]);
		//printf("]\n");
		for(int i = 0; i < m; i ++) {
			int src = vertices[i];
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
		// RMSE
		total_error = 0;
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
		printf("Iteration=%d: RMSE error = %f per edge\n", iter, sqrt(total_error/nnz));
	} while (iter < max_iters || total_error < epsilon);
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
}

