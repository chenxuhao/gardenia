// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include "timer.h"
#include <vector>
#include <algorithm>

// Compares with simple serial implementation that uses std::set_intersection
void TCVerifier(int m, IndexType *row_offsets, IndexType *column_indices, int test_total) {
	printf("Verifying...\n");
	int total = 0;
	vector<int> intersection;
	intersection.reserve(m);
	Timer t;
	t.Start();
	for (int src = 0; src < m; src ++) {
		IndexType row_begin = row_offsets[src];
		IndexType row_end = row_offsets[src + 1];
		for (IndexType offset = row_begin; offset < row_end; ++ offset) {
			IndexType dst = column_indices[offset];
			IndexType row_begin_dst = row_offsets[dst];
			IndexType row_end_dst = row_offsets[dst + 1];
			std::vector<int>::iterator new_end = set_intersection(column_indices + row_begin,
					column_indices + row_end,
					column_indices + row_begin_dst,
					column_indices + row_end_dst,
					intersection.begin());
			intersection.resize(new_end - intersection.begin());
			total += intersection.size();
		}
	}
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	total = total / 6;  // each triangle was counted 6 times
	if(total == test_total) printf("Correct\n");
	else printf("Wrong\n");
	printf("total=%d, test_total=%d\n", total, test_total);
	return;
}
