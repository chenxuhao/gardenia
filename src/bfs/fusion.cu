// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "fusion"
#include "bfs.h"
#include "worklistc.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#include "timer.h"
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
__device__ void expandByCta(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(in_queue.pop_id(id, vertex)) {
		size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(true) {
		if(size > BLOCK_SIZE)
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
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = depth;
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
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
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
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = depth;
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__device__ void process_vertex(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
	expandByCta(m, row_offsets, column_indices, dist, in_queue, out_queue, depth);
	expandByWarp(m, row_offsets, column_indices, dist, in_queue, out_queue, depth);
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int total_inputs = (*in_queue.d_index - 1) / (gridDim.x * blockDim.x) + 1;
	for (int id = tid; total_inputs > 0; id += blockDim.x * gridDim.x, total_inputs--) {
		int neighborsize = 0;
		int neighboroffset = 0;
		int scratch_offset = 0;
		int total_edges = 0;
		if(in_queue.pop_id(id, vertex)) {	  
			if(vertex != -1) {
				neighboroffset = row_offsets[vertex];
				neighborsize = row_offsets[vertex + 1] - neighboroffset;
			}
		}
		BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
		int done = 0;
		int neighborsdone = 0;
		while(total_edges > 0) {
			__syncthreads();
			int i;
			for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
				gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			}
			neighborsdone += i;
			scratch_offset += i;
			__syncthreads();
			int ncnt = 0;
			int dst = 0;
			int edge = gather_offsets[threadIdx.x];
			if(threadIdx.x < total_edges) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = depth;
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
			total_edges -= BLOCK_SIZE;
			done += BLOCK_SIZE;
		}
	}
}

__global__ void insert(int source, Worklist2 in_queue) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) in_queue.push(source);
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 in_queue, Worklist2 out_queue, int depth, GlobalBarrier gb) {
	Worklist2 *in;
	Worklist2 *out;
	Worklist2 *tmp;
	in = &in_queue; out = &out_queue;
	while(*in->d_index > 0) {
		depth ++;
		//if(blockIdx.x * blockDim.x + threadIdx.x == 0) printf("iteration %d: frontier_size = %d\n", depth, *in->d_index);
		process_vertex(m, row_offsets, column_indices, dist, *in, *out, depth);
		gb.Sync();
		tmp = in;
		in = out;
		out = tmp;
		*out->d_index = 0;
	}
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *in_degree, int *h_degree, DistT *h_dist) {
	DistT zero = 0;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));

	Worklist2 queue1(nnz), queue2(nnz);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	int nSM = 13;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	nSM = deviceProp.multiProcessorCount;
	int max_blocks = 5;
	max_blocks = maximum_residency(bfs_kernel, nthreads, 0);
	printf("max_blocks=%d, nSM=%d\n", max_blocks, nSM);
	nblocks = nSM * max_blocks;
	GlobalBarrierLifetime gb;
	gb.Setup(nblocks);
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	insert<<<1, 1>>>(source, *in_frontier);
	bfs_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_dist, *in_frontier, *out_frontier, 0, gb);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	int max_depth = 0;
	#pragma omp parallel for reduction(max : max_depth)
	for (int n = 0; n < m; n ++)
		if(h_dist[n] != MYINFINITY) max_depth = max(max_depth, h_dist[n]);
	printf("\titerations = %d.\n", max_depth+1);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
