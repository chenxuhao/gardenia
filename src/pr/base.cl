// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

inline void atomicAdd(volatile __global float *addr, float val) {
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

__kernel void contrib(int m, __global float *outgoing_contrib, __global int *degrees, __global float *scores) {
	int tid = get_global_id(0);
	if (tid < m) outgoing_contrib[tid] = scores[tid] / degrees[tid];
}

__kernel void pull(int m, __global int *row_offsets, __global int *column_indices, __global float *sums, __global float *outgoing_contrib) {
	int dst = get_global_id(0);
	if (dst < m) {
		int row_begin = row_offsets[dst];
		int row_end = row_offsets[dst+1];
		float incoming_total = 0;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int src = column_indices[offset];
			incoming_total += outgoing_contrib[src];
		}
		sums[dst] = incoming_total;
	}
}

#define BLK_SZ 256
inline float BlockReduce(float input) {
	int tx = get_local_id(0);
	__local float sdata[BLK_SZ];
	sdata[tx] = input;
	for (int c = BLK_SZ/2; c>0; c/=2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if(c > tx) sdata[tx] += sdata[tx+c];
	}
	//barrier(CLK_LOCAL_MEM_FENCE);
	return sdata[0];
}

__kernel void l1norm(int m, __global float *scores, __global float *sums, __global float *diff, float base_score, float kDamp) {
	int u = get_global_id(0);
	float local_diff = 0;
	if(u < m) {
		float new_score = base_score + kDamp * sums[u];
		local_diff += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	float block_sum = BlockReduce(local_diff);
	if(get_local_id(0) == 0) atomicAdd(diff, block_sum);
}

