// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

__kernel void hook(int m, __global int *row_offsets, __global int *column_indices,  __global int *comp, __global bool *changed) {
	int src = get_global_id(0);
	if(src < m) {
		int comp_src = comp[src];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = column_indices[offset];
			int comp_dst = comp[dst];
			if (comp_src == comp_dst) continue;
			int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
			int low_comp = comp_src + (comp_dst - high_comp);
			if (high_comp == comp[high_comp]) {
				*changed = true;
				comp[high_comp] = low_comp;
			}
		}
	}
}

__kernel void shortcut(int m, __global int *comp) {
	int src = get_global_id(0);
	if (src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

