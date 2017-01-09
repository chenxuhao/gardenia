#include "tc.h"
void TCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *degree, int *total) {
	int total_num = 0;
#pragma omp parallel for reduction(+ : total_num) schedule(dynamic, 64)
	for (int src = 0; src < m; src ++) {
		int row_begin_src = row_offsets[src];
		int row_end_src = row_offsets[src + 1]; 
		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			int dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst + 1];
			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				int dst_dst = column_indices[offset_dst];
				if (dst_dst > dst)
					break;
				while (column_indices[it] < dst_dst)
					it ++;
				if (dst_dst == column_indices[it])
					total_num ++;
			}
		} 
	}
	*total = total_num;
}
