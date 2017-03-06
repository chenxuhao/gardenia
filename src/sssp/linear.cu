// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "load-balance"
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
/*
[1] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, “Work-efficient
	parallel gpu methods for single-source shortest paths,” in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349–359, May 2014
*/
__global__ void initialize(int m, DistT *dist) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__device__ __forceinline__ void process_edge(int src, int edge, int *column_indices, DistT *weight, DistT *dist, Worklist2 &outwl) {
	int dst = column_indices[edge];
	DistT new_dist = dist[src] + weight[edge];
	if (new_dist < dist[dst]) {
		//atomicMin((unsigned *)&dist[dst], new_dist);
		atomicMin(&dist[dst], new_dist);
		outwl.push(dst);
	}
}
typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__device__ void expandByCta(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, Worklist2 &inwl, Worklist2 &outwl) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(inwl.pop_id(id, vertex)) {
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
			inwl.d_queue[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(sh_vertex, edge, column_indices, weight, dist, outwl);
			}
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
__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, Worklist2 &inwl, Worklist2 &outwl) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int vertex;
	if(inwl.pop_id(id, vertex)) {
		if (vertex != -1)
			size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = vertex;
			inwl.d_queue[id] = -1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(winner, edge, column_indices, weight, dist, outwl);
			}
		}
	}
}

__global__ void sssp_kernel(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, Worklist2 inwl, Worklist2 outwl) {
	//expandByCta(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	//expandByWarp(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLKSIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int src[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighborsize = 0;
	int neighboroffset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(inwl.pop_id(id, vertex)) {	  
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
			src[scratch_offset + i - done] = vertex;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			process_edge(src[threadIdx.x], edge, column_indices, weight, dist, outwl);
		}
		total_edges -= BLKSIZE;
		done += BLKSIZE;
	}
}

__global__ void insert(int source, Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(source);
	}
	return;
}

void SSSPSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist, int delta) {
	DistT zero = 0;
	int iter = 0;
	Timer t;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	//initialize <<<nblocks, nthreads>>> (d_dist, m);
	//CudaTest("initializing failed");

	int *d_row_offsets, *d_column_indices;
	DistT *d_weight;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(DistT), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));

	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	int max_blocks = maximum_residency(sssp_kernel, BLKSIZE, 0);
	printf("Solving, max_blocks=%d, nthreads=%d\n", max_blocks, nthreads);
	t.Start();
	insert<<<1, BLKSIZE>>>(source, *inwl);
	nitems = inwl->nitems();
	while(nitems > 0) {
		++ iter;
		nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
		//printf("iteration=%d, nblocks=%d, wlsz=%d\n", iter, nblocks, nitems);
		sssp_kernel<<<nblocks, BLKSIZE>>>(m, d_row_offsets, d_column_indices, d_weight, d_dist, *inwl, *outwl);
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	};
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());
	
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
