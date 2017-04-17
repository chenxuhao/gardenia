// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen 

Will return two latent vectors for users and items respectively.

This algorithm solves the matrix factorization problem for recommender 
systems using the SGD method described in [1].

[1] Yehuda Koren, Robert Bell and Chris Volinsky, Matrix factorization
	techniques for recommender systems,” IEEE Computer, 2009
*/
typedef float LatentT;
#define K 128 // dimension of the latent vector (number of features)
#define epsilon 1e-3f
#define max_iters 19

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step);
void SGDVerifier(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step);

static void print_latent_vector(int m, int n, LatentT *user_lv, LatentT *item_lv) {
	for (int i = 0; i < m; i ++) {
		printf("user_lv(%d): [ ", i);
		for (int j = 0; j < K; j++)
			printf("%.2f ", user_lv[i*K+j]);
		printf("]\n");
	}
	for (int i = 0; i < n; i ++) {
		printf("item_lv(%d): [ ", i);
		for (int j = 0; j < K; j++)
			printf("%.2f ", item_lv[i*K+j]);
		printf("]\n");
	}
}

