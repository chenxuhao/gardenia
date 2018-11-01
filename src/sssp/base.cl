// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

__kernel void init(int source, __global int *idx, __global int *queue, __global int *dists) {
	int tid = get_global_id(0);
	if (tid == 0) {
		dists[source] = 0;
		queue[*idx] = source;
		(*idx) ++;
	}
}

__kernel void sssp_step(int m, __global int *row_offsets, __global int *column_indices, __global unsigned *weight, __global int *dists, __global int *in_queue, __global int *out_queue, __global int *in_idx, __global int *out_idx) {
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
			int old_dist = dists[dst];
			int new_dist = dists[src] + weight[offset];
			if (new_dist < old_dist) {
				if (atomic_min(&dists[dst], new_dist) > new_dist) {
					int idx = atomic_add(out_idx, 1);
					out_queue[idx] = dst;
				}
			}
		}
	}
}
