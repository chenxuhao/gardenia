// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "linear"
#include "bfs.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#include "timer.h"

__global__ void initialize(DistT *dist, unsigned int m) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__device__ void expandByCta(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(in_queue.pop_id(id, vertex)) {
		size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(true) {
		if(size > BLKSIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1)
			break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = vertex;
			in_queue.d_queue[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			//int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = depth;
					out_queue.push(dst);
					//ncnt = 1;
				}
			}
			//out_queue.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define NUM_WARPS (BLKSIZE / WARP_SIZE)
__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int vertex;
	if(in_queue.pop_id(id, vertex)) {
		if (vertex != -1)
			size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = vertex;
			in_queue.d_queue[id] = -1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			//int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = depth;
					out_queue.push(dst);
					//ncnt = 1;
				}
			}
			//out_queue.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
		}
	}
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 in_queue, Worklist2 out_queue, int depth) {
	//expandByCta(m, row_offsets, column_indices, dist, in_queue, out_queue, depth);
	//expandByWarp(m, row_offsets, column_indices, dist, in_queue, out_queue, depth);
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLKSIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(in_queue.pop_id(id, vertex)) {
		if(vertex != -1) {
			neighbor_offset = row_offsets[vertex];
			neighbor_size = row_offsets[vertex+1] - neighbor_offset;
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
		//int ncnt = 0;
		int dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			assert(dst < m);
			if(dist[dst] == MYINFINITY) {
				dist[dst] = depth;
				//ncnt = 1;
				out_queue.push(dst);
			}
		}
		//out_queue.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
		total_edges -= BLKSIZE;
		done += BLKSIZE;
	}
}

__global__ void insert(int source, Worklist2 in_queue) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		in_queue.push(source);
	}
	return;
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	DistT zero = 0;
	int iter = 0;
	Timer t;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));

	//initialize <<<nblocks, nthreads>>> (m, d_dist);
	//CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 queue1(nnz), queue2(nnz);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int nitems = 1;
	t.Start();
	insert<<<1, BLKSIZE>>>(source, *in_frontier);
	nitems = in_frontier->nitems();
	do {
		++ iter;
		nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
		//printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iter, nblocks, BLKSIZE, nitems);
		bfs_kernel<<<nblocks, BLKSIZE>>>(m, d_row_offsets, d_column_indices, d_dist, *in_frontier, *out_frontier, iter);
		CudaTest("solving failed");
		nitems = out_frontier->nitems();
		Worklist2 *tmp = in_frontier;
		in_frontier = out_frontier;
		out_frontier = tmp;
		out_frontier->reset();
	} while(nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
