#include "common.h"
void BCSolver(int m, int nnz, int *row_offsets, int *column_indices, ScoreT *scores, int device);
void BCVerifier(int m, int *row_offsets, int *column_indices, int num_iters, ScoreT *scores_to_test);
