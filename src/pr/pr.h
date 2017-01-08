#include "common.h"
#define EPSILON 0.001
#define MAX_ITER 30
void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *score);
