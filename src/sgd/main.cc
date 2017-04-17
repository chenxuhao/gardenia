// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "graph_io.h"
static ScoreT lambda = 0.05; // regularization_factor
static ScoreT step = 0.002; // learning_rate

int main(int argc, char *argv[]) {
	printf("Stochastic Gradient Descent by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	if (argc > 2) lambda = atof(argv[2]);
	if (argc > 3) step = atof(argv[3]);
	printf("regularization_factor=%f, learning_rate=%f\n", lambda, step);
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false, false, false, false, false);

	LatentT *h_user_lv = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *h_item_lv = (LatentT *)malloc(n * K * sizeof(LatentT));
	LatentT *init_user_lv = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *init_item_lv = (LatentT *)malloc(n * K * sizeof(LatentT));
	ScoreT *h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));
	srand(0);
	for (int i = 0; i < m; i++) {
		unsigned r = i;
		for (int j = 0; j < K; j++)
			init_user_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	for (int i = 0; i < n; i++) {
		unsigned r = i + m;
		for (int j = 0; j < K; j++)
			init_item_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	for (int i = 0; i < m * K; i++) h_user_lv[i] = init_user_lv[i];
	for (int i = 0; i < n * K; i++) h_item_lv[i] = init_item_lv[i];

	for (int i = 0; i < nnz; i++) h_rating[i] = (ScoreT)h_weight[i];
	SGDSolver(m, n, nnz, h_row_offsets, h_column_indices, h_rating, h_user_lv, h_item_lv, lambda, step);
	SGDVerifier(m, n, nnz, h_row_offsets, h_column_indices, h_rating, init_user_lv, init_item_lv, lambda, step);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_user_lv);
	free(h_item_lv);
	free(h_rating);
	return 0;
}
