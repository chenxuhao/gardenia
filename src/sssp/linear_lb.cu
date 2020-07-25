// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "sssp.h"
#include "gbar.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__device__ __forceinline__ void process_edge(int src, int edge, 
                                             const VertexId *column_indices, 
                                             DistT *weight, DistT *dist, 
                                             Worklist2 &outwl) {
	int dst = column_indices[edge];
	DistT new_dist = dist[src] + weight[edge];
	if (new_dist < dist[dst]) {
		atomicMin(&dist[dst], new_dist);
		outwl.push(dst);
	}
}

__device__ void expandByCta(int m, const uint64_t *row_offsets, 
                            const VertexId *column_indices, 
                            DistT *weight, DistT *dist, 
                            Worklist2 &inwl, Worklist2 &outwl) {
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
		if(size > BLOCK_SIZE)
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
			int dst = 0;
			int ncnt = 0;
			if(i < neighbor_size) {
				int offset = row_begin + i;
				dst = column_indices[offset];
				DistT new_dist = dist[sh_vertex] + weight[offset];
				if (new_dist < dist[dst]) {
					atomicMin(&dist[dst], new_dist);
					ncnt = 1;
				}
			}
			outwl.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const uint64_t *row_offsets, 
                                             const VertexId *column_indices,
                                             DistT *weight, DistT *dist, 
                                             Worklist2 &inwl, Worklist2 &outwl) {
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
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
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

__global__ void sssp_kernel(int m, const uint64_t *row_offsets, 
                            const VertexId *column_indices,
                            DistT *weight, DistT *dist, 
                            Worklist2 inwl, Worklist2 outwl) {
	expandByCta(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	expandByWarp(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLOCK_SIZE;
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
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
}

void SSSPSolver(Graph &g, int source, DistT *h_weight, DistT *h_dist, int delta) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	DistT zero = 0;
	DistT *d_weight;
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	Worklist2 wl1(nnz), wl2(nnz);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	printf("Launching CUDA SSSP solver (block_size = %d) ...\n", nthreads);

	Timer t;
	t.Start();
	insert<<<1, 1>>>(source, *inwl);
	nitems = inwl->nitems();
	while(nitems > 0) {
		++ iter;
		nblocks = (nitems + BLOCK_SIZE - 1) / BLOCK_SIZE; 
		//printf("iteration %d: frontier_size = %d\n", iter, nitems);
		sssp_kernel<<<nblocks, BLOCK_SIZE>>>(m, d_row_offsets, d_column_indices, d_weight, d_dist, *inwl, *outwl);
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	};
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	
	printf("\titerations = %d.\n", iter);
	printf("\truntime [cuda_linear_lb] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}

