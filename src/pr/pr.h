#include "common.h"
#define EPSILON 0.001
#define MAX_ITER 19
void PRSolver(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree, ScoreT *d_score);
