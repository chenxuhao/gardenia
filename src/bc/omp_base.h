#include <omp.h>
#include "common.h"
void BCSolver(int m, int nnz, int *row_offsets, int *column_indices, ScoreT *scores, int device);
