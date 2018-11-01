// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

__kernel void init(int m, __global int *depths, __global int *path_counts, __global float *deltas) {
	int id = get_global_id(0);
	if (id < m) {
		depths[id] = -1;
		path_counts[id] = 0;
		deltas[id] = 0;
	}
}

__kernel void insert(int src, __global int *idx, __global int *queue, __global int *path_counts, __global int *depths) {
	int id = get_global_id(0);
	if (id == 0) {
		depths[src] = 0;
		path_counts[src] = 1;
		queue[*idx] = src;
		(*idx) ++;
	}
}

__kernel void bc_forward(__global int *row_offsets, __global int *column_indices, __global int *path_counts, __global int *depths, int depth, __global int *in_queue, __global int *out_queue, __global int *in_idx, __global int *out_idx) {
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
			if((depths[dst] == -1) && (atomic_cmpxchg(&depths[dst], -1, depth)) == -1) {
				int index = atomic_add(out_idx, 1);
				out_queue[index] = dst;
			}
			//barrier(CLK_GLOBAL_MEM_FENCE);
			if(depths[dst] == depth) {
				atomic_add(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

__kernel void bc_reverse(int num, __global int *row_offsets, __global int *column_indices, int start, __global int *queue, __global float *scores, __global int *path_counts, __global int *depths, int depth, __global float *deltas) {
	int tid = get_global_id(0);
	int src;
	if(tid < num) {
		int src = queue[start + tid];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				deltas[src] += (1.0 + deltas[dst]) * path_counts[src] / path_counts[dst];
			}
		}
		scores[src] += deltas[src];
	}
}

__kernel void push_frontier(int nitems, __global int *in_queue, __global int *queue, int len) {
	int tid = get_global_id(0);
	int vertex;
	int flag = 0;
	if (tid < nitems) {
		vertex = in_queue[tid];
		flag = 1;
	}
	if (flag == 1) queue[len + tid] = vertex;
}

__kernel void bc_normalize(int m, __global float *scores, float max_score) {
	int tid = get_global_id(0);
	if (tid < m) scores[tid] = scores[tid] / max_score;
}
/*
inline void atomicMax(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = max(prevVal.floatVal, operand);
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
*/
inline void atomicMax(volatile __global float *addr, float val) {
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = max(expected.f32, val);
		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

#define BLK_SZ 256
inline float BlockReduce(float input) {
	int tx = get_local_id(0);
	__local float sdata[BLK_SZ];
	sdata[tx] = input;
	for (int c = BLK_SZ/2; c>0; c/=2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if(c > tx) sdata[tx] = (sdata[tx] > sdata[tx+c]) ? sdata[tx] : sdata[tx+c];
	}
	//barrier(CLK_LOCAL_MEM_FENCE);
	return sdata[0];
}

__kernel void max_element(int m, __global float *scores, __global float *max_score) {
	int id = get_global_id(0);
	float block_max = BlockReduce(scores[id]);
	if(get_local_id(0) == 0) atomicMax(max_score, block_max);
}

