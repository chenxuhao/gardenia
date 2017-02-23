// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "timer.h"
void SGDVerifier(int m, int *row_offsets, int *column_indices, ScoreT *test_rating) {
	printf("Verifying...\n");
	Timer t;
	t.Start();
	t.Stop();
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
}

