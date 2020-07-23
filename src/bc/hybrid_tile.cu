// Copyright (c) 2016, Xuhao Chen
#include "bc.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <vector>
#include <cub/cub.cuh>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#define GPU_SEGMENTING
#include "segmenting.h"
#define BC_VARIANT "hybrid_tile"
#define USE_PULL

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;
typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduceInt;

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__global__ void initialize(int m, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) depths[id] = -1;
}

__global__ void insert(Worklist2 in_queue, int src, int *path_counts, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		in_queue.push(src);
		path_counts[src] = 1;
		depths[src] = 0;
	}
	return;
}

__global__ void push_frontier(Worklist2 in_queue, int *queue, int queue_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if (in_queue.pop_id(tid, vertex)) {
		queue[queue_len+tid] = vertex;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

__global__ void set_front(int source, int *front) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) front[source] = 1;
}

__global__ void BitmapToQueue(int m, int *bm, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m && bm[tid]) queue.push(tid);
}

__global__ void QueueToBitmap(int num, Worklist2 queue, int *bm) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		int src;
		if (queue.pop_id(tid, src)) bm[src] = 1;
	}
}

__global__ void FrontierToBitmap(int num, int *frontier, int *bm) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) bm[frontier[tid]] = 1;
}

__device__ __forceinline__ void process_edge(int value, int depth, int dst, const int *degrees, int *depths, int *path_counts, int *scout_count, Worklist2 &out_queue) {
	if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
		assert(out_queue.push(dst));
		atomicAdd(scout_count, __ldg(degrees+dst));
	}
	if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
}

__global__ void forward_base(const IndexT *row_offsets, const IndexT *column_indices, const int *degrees, int *depths, int *path_counts, int *scout_count, int depth, Worklist2 in_queue, Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
				assert(out_queue.push(dst));
				atomicAdd(scout_count, __ldg(degrees+dst));
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

__device__ __forceinline__ void expandByCta(const IndexT *row_offsets, const IndexT *column_indices, const int *degrees, int *depths, int *path_counts, int *scout_count, int depth, Worklist2 &in_queue, Worklist2 &out_queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_src;
	owner = -1;
	int size = 0;
	int src;
	if(in_queue.pop_id(id, src))
		size = row_offsets[src+1] - row_offsets[src];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_src = src;
			in_queue.d_queue[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_src];
		int row_end = row_offsets[sh_src+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		int value = path_counts[sh_src];
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int dst = 0;
			int ncnt = 0;
			int offset = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[offset];
				if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1))
					ncnt = 1;
				if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__device__ __forceinline__ void expandByWarp(const IndexT *row_offsets, const IndexT *column_indices, const int *degrees, int *depths, int *path_counts, int *scout_count, int depth, Worklist2 &in_queue, Worklist2 &out_queue) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int src;
	if(in_queue.pop_id(id, src))
		if (src != -1) size = row_offsets[src+1] - row_offsets[src];
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			in_queue.d_queue[id] = -1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_src[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int value = path_counts[winner];
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			if(i < neighbor_size) {
				int offset = row_begin + i;
				int dst = column_indices[offset];
				process_edge(value, depth, dst, degrees, depths, path_counts, scout_count, out_queue);
			}
		}
	}
}

__global__ void forward_lb(const IndexT *row_offsets, const IndexT *column_indices, const int *degrees, int *depths, int *path_counts, int *scout_count, int depth, Worklist2 in_queue, Worklist2 out_queue) {
	//expandByCta(row_offsets, column_indices, degrees, depths, path_counts, scout_count, depth, in_queue, out_queue);
	//expandByWarp(row_offsets, column_indices, degrees, depths, path_counts, scout_count, depth, in_queue, out_queue);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int src;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	__shared__ int values[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(in_queue.pop_id(tid, src)) {
		if(src != -1) {
			neighbor_offset = row_offsets[src];
			neighbor_size = row_offsets[src+1] - neighbor_offset;
			values[tx] = path_counts[src];
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			srcsrc[scratch_offset + i - done] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			process_edge(values[srcsrc[tx]], depth, dst, degrees, depths, path_counts, scout_count, out_queue);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void forward_topo(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *next, int depth) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m && depths[src] == depth-1) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		int value = path_counts[src];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				next[dst] = 1;
				atomicAdd(&path_counts[dst], value);
			}
		}
	}
}

__global__ void forward_topo_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *next, int depth) {
	//topo_expand_cta(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	//topo_expand_warp(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	__shared__ int values[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (src < m && depths[src] == depth-1) {
		neighbor_offset = row_offsets[src];
		neighbor_size = row_offsets[src+1] - neighbor_offset;
		values[tx] = path_counts[src];
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			srcsrc[scratch_offset + i - done] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				next[dst] = 1;
				atomicAdd(&path_counts[dst], values[srcsrc[tx]]);
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__device__ __forceinline__ void topo_lb_warp(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, const int *depths, int *path_counts, int *next, int depth, int *processed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int src;
	if(tid < m && !processed[tid]) {
		src = idx_map[tid];
		if (depths[src] == depth-1) {
			size = row_offsets[tid+1] - row_offsets[tid];
		}
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			processed[src] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_src[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int value = path_counts[winner];
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[offset];
				if (__ldg(depths+dst) == -1) {
					next[dst] = 1;
					atomicAdd(&path_counts[dst], value);
				}
			}
		}
	}
}

__global__ void forward_topo_lb_tiled(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, const int *depths, int *path_counts, int *next, int depth, int *front) {
	//topo_lb_cta(m, row_offsets, column_indices, idx_map, depths, path_counts, next, depth, front);
	//topo_lb_warp(m, row_offsets, column_indices, idx_map, depths, path_counts, next, depth, front);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	__shared__ int values[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (tid < m) {
		int src = idx_map[tid];
		if (depths[src] == depth-1) {
			neighbor_offset = row_offsets[tid];
			neighbor_size = row_offsets[tid+1] - neighbor_offset;
			values[tx] = path_counts[src];
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			srcsrc[scratch_offset + i - done] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				next[dst] = 1;
				atomicAdd(&path_counts[dst], values[srcsrc[tx]]);
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void forward_push(const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, Worklist2 in_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				visited[dst] = 1;
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

__global__ void forward_push_lb(const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, Worklist2 in_queue) {
	//push_expand_cta(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	//push_expand_warp(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int src;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	__shared__ int values[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(in_queue.pop_id(tid, src)) {
		if(src != -1) {
			neighbor_offset = row_offsets[src];
			neighbor_size = row_offsets[src+1] - neighbor_offset;
			values[tx] = path_counts[src];
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			srcsrc[scratch_offset + i - done] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				visited[dst] = 1;
				atomicAdd(&path_counts[dst], values[srcsrc[tx]]);
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__device__ __forceinline__ void pull_expand_cta(int m, const IndexT *row_offsets, const IndexT *column_indices,const int *idx_map, const int *depths, const int *path_counts, int *partial_counts, int *next, int depth, int *processed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduceInt::TempStorage temp_storage;
	__shared__ int owner;
	__shared__ int sh_dst;
	__shared__ int is_next;
	int size = 0;
	owner = -1;
	is_next = 0;
	if(tid < m) {
		int dst = idx_map[tid];
		if (depths[dst] == -1)
			size = row_offsets[tid+1] - row_offsets[tid];
	}
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_dst= tid;
			processed[tid] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_dst];
		int row_end = row_offsets[sh_dst+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		int sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				int src = column_indices[offset];
				if(depths[src] == depth-1) {
					is_next = 1;
					sum += path_counts[src];
					//sum += __ldg(path_counts+src);
				}
			}
		}
		int block_sum = BlockReduceInt(temp_storage).Sum(sum);
		if(threadIdx.x == 0) {
			next[sh_dst] = is_next;
			partial_counts[sh_dst] = block_sum;
		}
	}
}

__device__ __forceinline__ void pull_expand_warp(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, const int *depths, const int *path_counts, int *partial_counts, int *next, int depth, int *processed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_dst[NUM_WARPS];
	__shared__ int sdata[BLOCK_SIZE + 16];
	int size = 0;
	owner[warp_id] = -1;
	if(tid < m && !processed[tid]) {
		int dst = idx_map[tid];
		if (depths[dst] == -1)
			size = row_offsets[tid+1] - row_offsets[tid];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_dst[warp_id] = tid;
			processed[tid] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_dst[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int sum = 0;
		int is_next = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int src = column_indices[edge];
				if(depths[src] == depth-1) {
					is_next = 1;
					sum += path_counts[src];
					//sum += __ldg(path_counts+src);
				}
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) {
			next[winner] = is_next;
			partial_counts[winner] = sdata[threadIdx.x];
		}
	}
}

__global__ void forward_pull_base(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, const int *depths, const int *path_counts, int *partial_counts, int *next, int depth, int *processed) {
	pull_expand_cta(m, row_offsets, column_indices, idx_map, depths, path_counts, partial_counts, next, depth, processed);
	pull_expand_warp(m, row_offsets, column_indices, idx_map, depths, path_counts, partial_counts, next, depth, processed);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m && !processed[id]) {
	//if (id < m) {
		int dst = idx_map[id];
		int is_next = 0;
		int sum = 0;
		if (depths[dst] == -1) {
			IndexT row_begin = row_offsets[id];
			IndexT row_end = row_offsets[id+1];
			for (IndexT offset = row_begin; offset < row_end; ++ offset) {
				IndexT src = column_indices[offset];
				if(depths[src] == depth-1) {
					is_next = 1;
					sum += path_counts[src];
					//sum += __ldg(path_counts+src);
				}
			}
		}
		//next[dst] = is_next;
		next[id] = is_next;
		partial_counts[id] = sum;
		//st_glb_cs<int>(sum, partial_counts+id);
	}
}

__global__ void forward_pull_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, const int *depths, const int *path_counts, int *partial_counts, int *next, int depth, int *processed) {
	pull_expand_cta(m, row_offsets, column_indices, idx_map, depths, path_counts, partial_counts, next, depth, processed);
	pull_expand_warp(m, row_offsets, column_indices, idx_map, depths, path_counts, partial_counts, next, depth, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int dst_idx[BLOCK_SIZE];
	__shared__ int sums[BLOCK_SIZE];
	__shared__ int is_next[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	dst_idx[tx] = 0;
	sums[tx] = 0;
	is_next[tx] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (tid < m && !processed[tid]) {
		int dst = idx_map[tid];
		if (depths[dst] == -1) {
			neighbor_offset = row_offsets[tid];
			neighbor_size = row_offsets[tid+1] - neighbor_offset;
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while (total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			dst_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int edge = gather_offsets[tx];
			int src = column_indices[edge];
			if(depths[src] == depth-1) {
				atomicAdd(&sums[dst_idx[tx]], __ldg(path_counts+src));
				is_next[dst_idx[tx]] = 1;
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if (tid < m && !processed[tid]) {
		partial_counts[tid] = sums[tx];
		next[tid] = is_next[tx];
	}
}

__global__ void merge_count(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, int** partial_counts, int *path_counts, int** partial_next, int *next) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ int scounts[RANGE_WIDTH];
	__shared__ int snext[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		scounts[tx + i] = 0;
		snext[tx + i] = 0;
	}
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				int sidx = gid%RANGE_WIDTH;
				int local_count = partial_counts[bid][lid];
				scounts[sidx] += local_count;
				int local_next = partial_next[bid][lid];
				snext[sidx] |= local_next;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m) {
			path_counts[global_id] += scounts[local_id];
			next[global_id] = snext[local_id];
		}
	}
}

__global__ void merge_next(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, int** partial_next, int *next) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ int snext[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE)
		snext[tx + i] = 0;
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				int local_next = partial_next[bid][lid];
				snext[gid%RANGE_WIDTH] |= local_next;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m)
			next[global_id] = snext[local_id];
	}
}

__global__ void update_pull(int *next, int *depths, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (next[tid] == 1) depths[tid] = depth;
}

__global__ void update_push(const int *status, const int *degrees, int *depths, Worklist2 queue, int *scout_count, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(status[tid] == 1) {
		depths[tid] = depth;
		queue.push(tid);
		atomicAdd(scout_count, __ldg(degrees+tid));
	}
}

__global__ void reverse_base(int num, const IndexT *row_offsets, const IndexT *column_indices, int start, int *frontiers, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		int src = frontiers[start + tid];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		int value = path_counts[src];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (depths[dst] == depth + 1) {
				delta_src += static_cast<ScoreT>(value) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		deltas[src]  = delta_src;
		scores[src] += delta_src;
	}
}

__global__ void reverse_topo(int m, const IndexT *row_offsets, const IndexT *column_indices, int *front, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m && front[src]) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		int value = path_counts[src];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (depths[dst] == depth + 1) {
				delta_src += static_cast<ScoreT>(value) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		deltas[src]  = delta_src;
		scores[src] += delta_src;
	}
}

__device__ __forceinline__ void reverse_expand_cta(int num, const IndexT *row_offsets, const IndexT *column_indices, int start, IndexT *frontiers, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int src = 0;
	int size = 0;
	__shared__ int owner;
	__shared__ int sh_src;
	owner = -1;
	if(tid < num) {
		src = frontiers[start + tid];
		size = row_offsets[src+1] - row_offsets[src];
	}
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_src = src;
			frontiers[start + tid] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_src];
		int row_end = row_offsets[sh_src+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		int count = path_counts[sh_src];
		ScoreT sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[offset];
				if(depths[dst] == depth + 1) {
					ScoreT value = static_cast<ScoreT>(count) /
						static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
					sum += value;
				}
			}
		}
		ScoreT delta_src = BlockReduce(temp_storage).Sum(sum);
		if(threadIdx.x == 0) {
			deltas[sh_src]  = delta_src;
			scores[sh_src] += delta_src;
		}
	}
}

__device__ __forceinline__ void reverse_expand_warp(int num, const IndexT *row_offsets, const IndexT *column_indices, int start, IndexT *frontiers, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];
	owner[warp_id] = -1;
	int size = 0;
	int src = -1;
	if(tid < num) {
		src = frontiers[start + tid];
		if(src != -1) {
			size = row_offsets[src+1] - row_offsets[src];
		}
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			frontiers[start + tid] = -1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_src[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int count = path_counts[winner];
		ScoreT sum = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[edge];
				if(depths[dst] == depth + 1) {
					ScoreT value = static_cast<ScoreT>(count) /
						static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
					sum += value;
				}
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) {
			deltas[winner]  = sdata[threadIdx.x];
			scores[winner] += sdata[threadIdx.x];
		}
	}
}

__global__ void reverse_lb(int num, const IndexT *row_offsets, const IndexT *column_indices, int start, IndexT *frontiers, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	reverse_expand_cta(num, row_offsets, column_indices, start, frontiers, scores, path_counts, depths, depth, deltas);
	reverse_expand_warp(num, row_offsets, column_indices, start, frontiers, scores, path_counts, depths, depth, deltas);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	//__shared__ int srcs[BLOCK_SIZE];
	__shared__ int idx[BLOCK_SIZE];
	__shared__ int sh_counts[BLOCK_SIZE];
	__shared__ ScoreT sh_deltas[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	//srcs[tx] = 0;
	idx[tx] = 0;
	sh_counts[tx] = 0;
	sh_deltas[tx] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	int src = -1;
	if(tid < num) {
		src = frontiers[start + tid];
		if(src != -1) {
			neighbor_offset = row_offsets[src];
			neighbor_size = row_offsets[src+1] - neighbor_offset;
			sh_counts[tx] = path_counts[src];
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			//srcs[j] = src;
			idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				ScoreT value = static_cast<ScoreT>(sh_counts[idx[tx]]) / 
					static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
				atomicAdd(&sh_deltas[idx[tx]], value); 
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if(src != -1) {
		deltas[src]  = sh_deltas[tx];
		scores[src] += sh_deltas[tx];
	}
}

__device__ __forceinline__ void reverse_topo_expand_cta(int m, const IndexT *row_offsets, const IndexT *column_indices, int *front, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ int owner;
	__shared__ int sh_src;
	int size = 0;
	owner = -1;
	if(src < m && front[src])
		size = row_offsets[src+1] - row_offsets[src];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_src = src;
			front[src] = 0;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_src];
		int row_end = row_offsets[sh_src+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		int count = path_counts[sh_src];
		ScoreT sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[offset];
				if(depths[dst] == depth + 1) {
					ScoreT value = static_cast<ScoreT>(count) /
						static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
					sum += value;
				}
			}
		}
		ScoreT delta_src = BlockReduce(temp_storage).Sum(sum);
		if(threadIdx.x == 0) {
			deltas[sh_src]  = delta_src;
			scores[sh_src] += delta_src;
		}
	}
}

__device__ __forceinline__ void reverse_topo_expand_warp(int m, const IndexT *row_offsets, const IndexT *column_indices, int *front, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];
	owner[warp_id] = -1;
	int size = 0;
	if(src < m && front[src])
		size = row_offsets[src+1] - row_offsets[src];
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			front[src] = 0;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_src[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		int count = path_counts[winner];
		ScoreT sum = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			if(i < neighbor_size) {
				int offset = row_begin + i;
				int dst = column_indices[offset];
				if(depths[dst] == depth + 1) {
					ScoreT value = static_cast<ScoreT>(count) /
						static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
					sum += value;
				}
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) {
			deltas[winner]  = sdata[threadIdx.x];
			scores[winner] += sdata[threadIdx.x];
		}
	}
}

__global__ void reverse_topo_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, int *front, ScoreT *scores, const int *path_counts, int *depths, int depth, ScoreT *deltas) {
	reverse_topo_expand_cta(m, row_offsets, column_indices, front, scores, path_counts, depths, depth, deltas);
	reverse_topo_expand_warp(m, row_offsets, column_indices, front, scores, path_counts, depths, depth, deltas);
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int idx[BLOCK_SIZE];
	__shared__ int sh_counts[BLOCK_SIZE];
	__shared__ ScoreT sh_deltas[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	idx[tx] = 0;
	sh_counts[tx] = 0;
	sh_deltas[tx] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(src < m && front[src]) {
		neighbor_offset = row_offsets[src];
		neighbor_size = row_offsets[src+1] - neighbor_offset;
		sh_counts[tx] = path_counts[src];
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				ScoreT value = static_cast<ScoreT>(sh_counts[idx[tx]]) / 
					static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
				atomicAdd(&sh_deltas[idx[tx]], value); 
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if(src < m && front[src]) {
		deltas[src]  = sh_deltas[tx];
		scores[src] += sh_deltas[tx];
	}
}

__global__ void reverse_topo_lb_tiled(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *idx_map, int *front, const int *path_counts, const int *depths, int depth, const ScoreT *deltas, ScoreT *partial_deltas) {
	//reverse_topo_expand_cta(m, row_offsets, column_indices, front, path_counts, depths, depth, deltas);
	//reverse_topo_expand_warp(m, row_offsets, column_indices, front, path_counts, depths, depth, deltas);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int idx[BLOCK_SIZE];
	__shared__ int sh_counts[BLOCK_SIZE];
	__shared__ ScoreT sh_deltas[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	idx[tx] = 0;
	sh_counts[tx] = 0;
	sh_deltas[tx] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	int src;
	if(tid < m) {
		src = idx_map[tid];
		if (front[src]) {
			neighbor_offset = row_offsets[tid];
			neighbor_size = row_offsets[tid+1] - neighbor_offset;
			sh_counts[tx] = path_counts[src];
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				ScoreT value = static_cast<ScoreT>(sh_counts[idx[tx]]) / 
					static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
				atomicAdd(&sh_deltas[idx[tx]], value); 
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if(tid < m) partial_deltas[tid] = sh_deltas[tx];
}

__global__ void update_scores(int num, ScoreT *deltas, int *frontiers, ScoreT *scores) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		int src = frontiers[tid];
		scores[src] += deltas[src];
	}
}

__global__ void merge_deltas(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ScoreT** partial_deltas, ScoreT *deltas) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ ScoreT sdeltas[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		sdeltas[tx + i] = 0;
	}
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				int sidx = gid%RANGE_WIDTH;
				ScoreT local_delta = partial_deltas[bid][lid];
				sdeltas[sidx] += local_delta;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m) {
			deltas[global_id] += sdeltas[local_id];
		}
	}
}

void BCSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *h_degrees, ScoreT *h_scores) {
	//print_device_info(0);
#ifdef USE_PULL
	segmenting(m, in_row_offsets, in_column_indices, NULL);
#else
	segmenting(m, out_row_offsets, out_column_indices, NULL);
#endif
/*
	int *d_in_row_offsets, *d_in_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
*/
	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (m - 1) / RANGE_WIDTH + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs);
	IndexT ** d_range_indices = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
#ifdef USE_PULL
	int ** d_partial_counts = (int**)malloc(num_subgraphs * sizeof(int*));
	int ** d_next = (int**)malloc(num_subgraphs * sizeof(int*));
#else
	ScoreT ** d_partial_deltas = (ScoreT**)malloc(num_subgraphs * sizeof(ScoreT*));
#endif

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_range_indices[bid], (num_ranges+1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_range_indices[bid], range_indices[bid], (num_ranges+1) * sizeof(IndexT), cudaMemcpyHostToDevice));
#ifdef USE_PULL
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_counts[bid], ms_of_subgraphs[bid] * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_next[bid], ms_of_subgraphs[bid] * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(d_partial_counts[bid], 0, ms_of_subgraphs[bid] * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(d_next[bid], 0, ms_of_subgraphs[bid] * sizeof(int)));
#else
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_deltas[bid], ms_of_subgraphs[bid] * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemset(d_partial_deltas[bid], 0, ms_of_subgraphs[bid] * sizeof(ScoreT)));
#endif
	}

	printf("copy host pointers to device\n");
	IndexT ** d_range_indices_ptr, **d_idx_map_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_range_indices_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_idx_map_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_range_indices_ptr, d_range_indices, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_idx_map_ptr, d_idx_map, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
#ifdef USE_PULL
	int **d_partial_counts_ptr, **d_next_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_counts_ptr, num_subgraphs * sizeof(int*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_next_ptr, num_subgraphs * sizeof(int*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_counts_ptr, d_partial_counts, num_subgraphs * sizeof(int*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_next_ptr, d_next, num_subgraphs * sizeof(int*), cudaMemcpyHostToDevice));
#else
	ScoreT **d_partial_deltas_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_deltas_ptr, num_subgraphs * sizeof(ScoreT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_deltas_ptr, d_partial_deltas, num_subgraphs * sizeof(ScoreT*), cudaMemcpyHostToDevice));
#endif
	//bool *d_processed;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(bool)));

	int *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores, *d_deltas;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMemset(d_scores, 0, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemset(d_deltas, 0, m * sizeof(ScoreT)));
	int *d_path_counts, *d_depths, *d_frontiers;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (m+1)));
	CUDA_SAFE_CALL(cudaMemset(d_path_counts, 0, m * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	int *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, m * sizeof(int)));
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	int *d_scout_count;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
	int *front, *next;
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));

	int depth = 0;
	int nitems = 1;
	int frontiers_len = 0;
	vector<int> depth_index;
	vector<int> direction;
	depth_index.push_back(0);
	Worklist2 wl1(m), wl2(m);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nthreads = BLOCK_SIZE;
	int mblocks = (m - 1) / nthreads + 1;
	int alpha = 15, beta = 18;
	int edges_to_check = nnz;
	int scout_count = h_degrees[source];
	//set_front<<<1, 1>>>(source, front);
	initialize<<<mblocks, nthreads>>>(m, d_depths);
	insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", mblocks, nthreads);

	Timer t;
	t.Start();
	do {
		if(scout_count > edges_to_check / alpha) {
			int awake_count, old_awake_count;
			QueueToBitmap<<<((nitems-1)/512+1), 512>>>(nitems, *inwl, front);
			awake_count = nitems;
			do {
				++ depth;
				int nblocks = (awake_count - 1) / nthreads + 1;
				push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
				frontiers_len += awake_count;
				depth_index.push_back(frontiers_len);
				direction.push_back(0);
				old_awake_count = awake_count;
				//printf("BU: iteration=%d, num_frontier=%d\n", depth, awake_count);
#ifdef USE_PULL
				for (int bid = 0; bid < num_subgraphs; bid ++) {
					int n_vertices = ms_of_subgraphs[bid];
					int bblocks = (n_vertices - 1) / nthreads + 1;
					CUDA_SAFE_CALL(cudaMemset(front, 0, n_vertices * sizeof(int)));
					forward_pull_lb<<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_depths, d_path_counts, d_partial_counts[bid], d_next[bid], depth, front);
					CudaTest("solving forward failed");
				}
				merge_count<<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_counts_ptr, d_path_counts, d_next_ptr, next);
				//merge_next<<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_next_ptr, next);
				CudaTest("solving merge failed");
#else
				//forward_topo_lb<<<mblocks, nthreads>>>(m, d_out_row_offsets, d_out_column_indices, d_depths, d_path_counts, next, depth);
				for (int bid = 0; bid < num_subgraphs; bid ++) {
					int n_vertices = ms_of_subgraphs[bid];
					int bblocks = (n_vertices - 1) / nthreads + 1;
					CUDA_SAFE_CALL(cudaMemset(front, 0, n_vertices * sizeof(int)));
					forward_topo_lb_tiled<<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], d_depths, d_path_counts, next, depth, front);
					CudaTest("solving forward failed");
				}
#endif
				update_pull<<<mblocks, nthreads>>>(next, d_depths, depth);
				awake_count = thrust::reduce(thrust::device, next, next + m, 0, thrust::plus<int>());
				int *temp = front;
				front = next;
				next = temp;
				inwl->reset();
				BitmapToQueue<<<((m-1)/512+1), 512>>>(m, front, *inwl);
				CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
			} while((awake_count >= old_awake_count) || (awake_count > m / beta));
			inwl->reset();
			BitmapToQueue<<<((m-1)/512+1), 512>>>(m, front, *inwl);
			scout_count = 1;
			nitems = inwl->nitems();
		} else {
			++ depth;
			nitems = inwl->nitems();
			//printf("TD: iteration=%d, num_frontier=%d\n", depth, nitems);
			CUDA_SAFE_CALL(cudaMemset(d_status, 0, m * sizeof(int)));
			edges_to_check -= scout_count;
			int nblocks = (nitems - 1) / nthreads + 1;
			push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
			frontiers_len += nitems;
			depth_index.push_back(frontiers_len);
			direction.push_back(1);
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			//if (nitems <512) {
				forward_lb<<<nblocks, nthreads>>>(d_out_row_offsets, d_out_column_indices, d_degrees, d_depths, d_path_counts, d_scout_count, depth, *inwl, *outwl);
			//} else {
			//	forward_push_lb<<<nblocks, nthreads>>>(d_out_row_offsets, d_out_column_indices, d_depths, d_path_counts, d_status, *inwl);
			//	update_push<<<mblocks, nthreads>>>(d_status, d_degrees, d_depths, *outwl, d_scout_count, depth);
			//}
			CudaTest("solving kernel forward failed");
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			nitems = outwl->nitems();
			Worklist2 *tmp = inwl;
			inwl = outwl;
			outwl = tmp;
			outwl->reset();
		}
	} while (nitems > 0);
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		if (direction[d]) {
			//printf("Data: depth=%d, frontier_size=%d\n", d, nitems);
			int nblocks = (nitems - 1) / nthreads + 1;
			reverse_lb<<<nblocks, nthreads>>>(nitems, d_out_row_offsets, d_out_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		} else {
			//printf("Topo: depth=%d, frontier_size=%d\n", d, nitems);
			CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
			FrontierToBitmap<<<((nitems-1)/512+1), 512>>>(nitems, d_frontiers+depth_index[d], front);
#ifdef USE_PULL
			reverse_topo_lb<<<mblocks, nthreads>>>(m, d_out_row_offsets, d_out_column_indices, front, d_scores, d_path_counts, d_depths, d, d_deltas);
#else
			for (int bid = 0; bid < num_subgraphs; bid ++) {
				int n_vertices = ms_of_subgraphs[bid];
				int bblocks = (n_vertices - 1) / nthreads + 1;
				reverse_topo_lb_tiled<<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], front, d_path_counts, d_depths, d, d_deltas, d_partial_deltas[bid]);
			}
			merge_deltas<<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_deltas_ptr, d_deltas);
			int nblocks = (nitems - 1) / nthreads + 1;
			update_scores<<<nblocks, nthreads>>>(nitems, d_deltas, d_frontiers+depth_index[d], d_scores);
#endif
		}
		CudaTest("solving kernel reverse failed");
	}
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	bc_normalize<<<mblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\tmax_score = %.6f.\n", h_max_score);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
}

