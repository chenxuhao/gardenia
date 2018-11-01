// Copyright (c) 2016, Xuhao Chen
#define BC_VARIANT "linear_lb"
#include "bc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"
#include "timer.h"
#include <vector>
#include <cub/cub.cuh>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void initialize(int m, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) depths[id] = -1;
}

__device__ __forceinline__ void process_edge(int src, int depth, int edge, int *column_indices, int *path_counts, int *depths, Worklist2 &out_queue) {
	int dst = column_indices[edge];
	if((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
		out_queue.push(dst);
	}
	if (depths[dst] == depth) {
		atomicAdd(&path_counts[dst], path_counts[src]);
	}
}

__device__ void expandByCta(int m, int *row_offsets, int *column_indices, int *path_counts, int *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
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
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(sh_vertex, depth, edge, column_indices, path_counts, depths, out_queue);
			}
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, int *path_counts, int *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
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
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(winner, depth, edge, column_indices, path_counts, depths, out_queue);
			}
		}
	}
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(int m, int *row_offsets, int *column_indices, int *path_counts, int *depths, int depth, Worklist2 in_queue, Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
/*
	if(in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
				assert(out_queue.push(dst));
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
*/
	expandByCta(m, row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	expandByWarp(m, row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(in_queue.pop_id(tid, src)) {
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
			srcsrc[scratch_offset + i - done] = src;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			process_edge(srcsrc[threadIdx.x], depth, edge, column_indices, path_counts, depths, out_queue);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__device__ __forceinline__ void process_edge_reverse(int src, int depth, int edge, int *column_indices, int *path_counts, int *depths, Worklist2 &out_queue) {
	//int dst = column_indices[edge];
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
/*
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < num) {
		int src = frontiers[start + tid];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				deltas[src] += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		scores[src] += deltas[src];
	}
//*/
///*
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction conditionals

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int index = warp_id; index < num; index += num_warps) {
		int src = frontiers[start + index];
		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[src + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_start = row_offsets[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[row+1];
		ScoreT sum = 0;
		for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				sum += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		// store local sum in shared memory
		sdata[threadIdx.x] = sum; __syncthreads();

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if (thread_lane == 0) {
			deltas[src] += sdata[threadIdx.x];
			scores[src] += deltas[src];
		}
	}
//*/
/*
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int srcsrc[SCRATCHSIZE];
	//__shared__ int owner[SCRATCHSIZE];
	__shared__ ScoreT delta_src[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	srcsrc[threadIdx.x] = 0;
	delta_src[threadIdx.x] = 0;
	//owner[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	int vertex;
	if(tid < num) {
		vertex = frontiers[start + tid];
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
			srcsrc[scratch_offset + i - done] = vertex;
			owner[scratch_offset + i - done] = threadIdx.x;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			int dst = column_indices[edge];
			//int owner_tid = owner[threadIdx.x];
			if(depths[dst] == depth + 1) {
				delta_src[owner_tid] += static_cast<ScoreT>(path_counts[srcsrc[threadIdx.x]]) / 
				//deltas[srcsrc[threadIdx.x]] += static_cast<ScoreT>(path_counts[srcsrc[threadIdx.x]]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
			deltas[srcsrc[threadIdx.x]] +=
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if(vertex != -1) {
		scores[vertex] += deltas[vertex];
	}
//*/
}

__global__ void insert(Worklist2 inwl, int src, int *path_counts, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(src);
		path_counts[src] = 1;
		depths[src] = 0;
	}
	return;
}

__global__ void push_frontier(Worklist2 inwl, int *queue, int queue_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(inwl.pop_id(tid, vertex)) {
		queue[queue_len+tid] = vertex;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

void BCSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
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

	int depth = 0;
	int nitems = 1;
	int frontiers_len = 0;
	vector<int> depth_index;
	depth_index.push_back(0);
	Worklist2 wl1(m), wl2(m);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, d_depths);
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(bc_reverse, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
	do {
		nblocks = (nitems - 1) / nthreads + 1;
		push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
		frontiers_len += nitems;
		depth_index.push_back(frontiers_len);
		//printf("Forward: depth=%d, frontire_size=%d\n", depth, nitems);
		depth++;
		bc_forward<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_path_counts, d_depths, depth, *inwl, *outwl);
		CudaTest("solving kernel bc_forward failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		//nblocks = (nitems - 1) / nthreads + 1;
		nblocks = std::min(max_blocks, DIVIDE_INTO(nitems, WARPS_PER_BLOCK));
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
		bc_reverse<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving kernel bc_reverse failed");
	}
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	nthreads = 512;
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

