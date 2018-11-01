// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

__kernel void ordered_count(int m, __global int *row_offsets, __global int *column_indices, __global int *total) {
	int src = get_global_id(0);
	int num = 0;
	if(src < m) {
		int row_begin_src = row_offsets[src];
		int row_end_src = row_offsets[src+1];
		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			int dst = column_indices[offset_src];
			if (dst > src)
				break;
			int it = row_begin_src;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst+1];
			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				int dst_dst = column_indices[offset_dst];
				if(dst_dst > dst)
					break;
				while (column_indices[it] < dst_dst)
					it ++;
				if (dst_dst == column_indices[it])
					num ++;
			}
		}
		atomic_add(total, num);
	}
}
