// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "bfs.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define BFS_VARIANT "linear_lb"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__device__ __forceinline__ void process_edge(int depth, int edge, 
                                             const IndexT *column_indices, 
                                             DistT *depths, Worklist2 &out_queue) {
	int dst = column_indices[edge];
	//if (depths[dst] > depth) {
	if (depths[dst] == MYINFINITY) {
		depths[dst] = depth;
		out_queue.push(dst);
	}
}

__device__ __forceinline__ void expandByCta(int m, const uint64_t *row_offsets, 
                                            const IndexT *column_indices, 
                                            DistT *depths, Worklist2 &in_queue, 
                                            Worklist2 &out_queue, int depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(in_queue.pop_id(id, vertex)) {
		size = row_offsets[vertex+1] - row_offsets[vertex];
	}
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = vertex;
			in_queue.d_queue[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			int dst = 0;
			int ncnt = 0;
			if(i < neighbor_size) {
				// TODO: push() doesn't work for expandByCta
				//process_edge(depth, edge, column_indices, depths, out_queue);
				///*
				dst = column_indices[edge];
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					ncnt = 1;
				}
				//*/
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

__device__ __forceinline__ void expandByWarp(int m, const uint64_t *row_offsets, 
                                             const IndexT *column_indices, 
                                             DistT *depths, Worklist2 &in_queue, 
                                             Worklist2 &out_queue, int depth) {
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
			size = row_offsets[vertex+1] - row_offsets[vertex];
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
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			//int ncnt = 0;
			//int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(depth, edge, column_indices, depths, out_queue);
				/*
				dst = column_indices[edge];
				//assert(dst < m);
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					ncnt = 1;
				}
				//*/
			}
			//out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__global__ void bfs_kernel(int m, const uint64_t *row_offsets, 
                           const IndexT *column_indices, 
                           DistT *depths, Worklist2 in_queue, 
                           Worklist2 out_queue, int depth) {
	expandByCta(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	expandByWarp(m, row_offsets, column_indices, depths, in_queue, out_queue, depth);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLOCK_SIZE;
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
		//int dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			process_edge(depth, edge, column_indices, depths, out_queue);
			/*
			dst = column_indices[edge];
			//assert(dst < m);
			if(depths[dst] == MYINFINITY) {
				depths[dst] = depth;
				ncnt = 1;
			}
			//*/
		}
		//out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void insert(int source, Worklist2 in_queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) in_queue.push(source);
	return;
}

void BFSSolver(Graph &g, int source, DistT *h_depths) {
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
  DistT * d_depths;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
  CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, m * sizeof(DistT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Worklist2 queue1(nnz), queue2(nnz);
  Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
  int iter = 0;
  int nitems = 1;
  int nthreads = BLOCK_SIZE;
  int nblocks = (m - 1) / nthreads + 1;
  printf("Launching CUDA BFS solver (%d threads/CTA) ...\n", nthreads);

  Timer t;
  t.Start();
  insert<<<1, nthreads>>>(source, *in_frontier);
  nitems = in_frontier->nitems();
  do {
    ++ iter;
    nblocks = (nitems + nthreads - 1) / nthreads; 
    //printf("iteration=%d, frontier_size=%d\n", iter, nitems);
    bfs_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, 
        d_depths, *in_frontier, *out_frontier, iter);
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
  printf("\truntime [cuda_linear_lb] = %f ms.\n", t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_row_offsets));
  CUDA_SAFE_CALL(cudaFree(d_column_indices));
  CUDA_SAFE_CALL(cudaFree(d_depths));
  return;
}

