// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

#define MYINFINITY	1000000000

__kernel void init(int source, __global int *idx, __global int *queue, __global int *depths) {
	int tid = get_global_id(0);
	if (tid == 0) {
		depths[source] = 0;
		queue[*idx] = source;
		(*idx) ++;
	}
}

__kernel void bfs_step(int m, __global int *row_offsets, __global int *column_indices, __global int *depths, __global int *in_queue, __global int *out_queue, __global int *in_idx, __global int *out_idx) {
	int tid = get_global_id(0);
	int src, flag = 0;
	if (tid < (*in_idx)) {
		src = in_queue[tid];
		flag = 1;
	}
	if (flag) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == MYINFINITY) && (atomic_cmpxchg(&depths[dst], MYINFINITY, depths[src]+1) == MYINFINITY)) {
				int idx = atomic_add(out_idx, 1);
				out_queue[idx] = dst;
			}
		}
	}
}
