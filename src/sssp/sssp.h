#include "common.h"
const DistT kDistInf = numeric_limits<DistT>::max()/2;
void SSSPSolver(int m, int nnz, int *d_row_offsets, int *d_column_indices, DistT *d_weight, DistT *d_dist);
void SSSPVerifier(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist);
