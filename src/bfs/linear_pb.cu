// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define BFS_VARIANT "linear_pb"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__device__ __forceinline__ void expandByCtaStep(int m, const IndexT *row_offsets, const IndexT *column_indices, DistT *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	__shared__ int owner;
	__shared__ int sh_src;
	owner = -1;
	int size = 0;
	if(in_queue.pop_id(id, src)) {
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
			in_queue.d_queue[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_src];
		int row_end = row_offsets[sh_src+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			int dst = 0;
			int ncnt = 0;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarpStep(int m, const IndexT *row_offsets, const IndexT *column_indices, DistT *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int src;
	if(in_queue.pop_id(id, src)) {
		if (src != -1)
			size = row_offsets[src+1] - row_offsets[src];
	}
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
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__global__ void step_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, DistT *depths, Worklist2 in_queue, Worklist2 out_queue, int depth) {
	expandByCtaStep(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	expandByWarpStep(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(in_queue.pop_id(id, src)) {
		if(src != -1) {
			neighbor_offset = row_offsets[src];
			neighbor_size = row_offsets[src+1] - neighbor_offset;
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
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int ncnt = 0;
		int dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			if(depths[dst] == MYINFINITY) {
				depths[dst] = depth;
				ncnt = 1;
			}
		}
		out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void push_base(int m, const IndexT *row_offsets, const IndexT *column_indices, bool *visited, Worklist2 in_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			visited[dst] = 1;
		}
	}
}

__device__ __forceinline__ void expandByCta(int n, const IndexT *row_offsets, const IndexT *column_indices, bool *visited, Worklist2 &in_queue) {
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
			assert(id < n);
			in_queue.d_queue[id] = -1;
			owner = -1;
			size = 0;
			//if(threadIdx.x == 0) printf("src %d selected\n", sh_src);
		}
		__syncthreads();
		int row_begin = row_offsets[sh_src];
		int row_end = row_offsets[sh_src+1];
		int neighbor_size = row_end - row_begin;
		//if(threadIdx.x == 0) printf("src %d with %d nerghbors\n", sh_src, neighbor_size);
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[edge];
				visited[dst] = 1;
			}
		}
	}
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *row_offsets, const IndexT *column_indices, bool *visited, Worklist2 &in_queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int src;
	if(in_queue.pop_id(id, src)) {
		if (src != -1)
			size = row_offsets[src+1] - row_offsets[src];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			in_queue.d_queue[id] = -1;
			owner[warp_id] = -1;
			size = 0;
			//if(lane_id == 0) printf("src %d selected\n", sh_src[warp_id]);
		}
		int winner = sh_src[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		//if(lane_id == 0) printf("src %d with %d nerghbors\n", winner, neighbor_size);
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = column_indices[edge];
				visited[dst] = 1;
			}
		}
	}
}

__global__ void push_lb(int num, const IndexT *row_offsets, const IndexT *column_indices, bool *visited, Worklist2 in_queue) {
	expandByCta(num, row_offsets, column_indices, visited, in_queue);
	expandByWarp(num, row_offsets, column_indices, visited, in_queue);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	int src;
	if(in_queue.pop_id(id, src)) {
		if(src != -1) {
			neighbor_offset = row_offsets[src];
			neighbor_size = row_offsets[src+1] - neighbor_offset;
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
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			int dst = column_indices[edge];
			visited[dst] = 1;
			//printf("dst %d\n", dst);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void update(int m, DistT *depths, bool *visited, Worklist2 queue, int depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(depths[id] == MYINFINITY && visited[id]) {
			depths[id] = depth;
			queue.push(id);
		}
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
}

void BFSSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *h_row_offsets, IndexT *h_column_indices, int *in_degrees, int *h_degrees, DistT *h_dists) {
	//print_device_info(0);
	DistT zero = 0;
	bool one = 1;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_depths;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_dists, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	bool *d_visited;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(&d_visited[source], &one, sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Worklist2 queue1(m), queue2(m);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int iter = 0;
	int nitems = 1;
	int nthreads = BLOCK_SIZE;
	int mblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA BFS solver (%d threads/CTA) ...\n", nthreads);

	Timer t;
	t.Start();
	insert<<<1, nthreads>>>(source, *in_frontier);
	nitems = in_frontier->nitems();
	do {
		++ iter;
		int nblocks = (nitems - 1) / nthreads + 1;
		printf("iteration %d: frontier_size = %d\n", iter, nitems);
		//if (nitems < 1024)
		if (0)
			//step <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_depths, *in_frontier, *out_frontier);
			step_lb <<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_depths, *in_frontier, *out_frontier, iter);
		else {
			//push_base <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_visited, *in_frontier);
			push_lb <<<nblocks, nthreads>>> (nitems, d_row_offsets, d_column_indices, d_visited, *in_frontier);
			CudaTest("solving push failed");
			update <<<mblocks, nthreads>>> (m, d_depths, d_visited, *out_frontier, iter);
			CudaTest("solving update failed");
		}
		nitems = out_frontier->nitems();
		Worklist2 *tmp = in_frontier;
		in_frontier = out_frontier;
		out_frontier = tmp;
		out_frontier->reset();
		printf("next iteration frontier_size = %d\n", nitems);
		if(iter == 2) return;
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	return;
}
