// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#define GPU_SEGMENTING
#include "segmenting.h"
#define BFS_VARIANT "hybrid_tile"
#define ENABLE_TILE

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__global__ void bottom_up_base(int m, const IndexT *row_offsets, const IndexT *column_indices, DistT *depths, int *front, int *next, int depth) {
	//bu_expand_warp(m, row_offsets, column_indices, depths, front, next, depth);
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if(dst < m && depths[dst] == MYINFINITY) { // not visited
		int row_begin = row_offsets[dst];
		int row_end = row_offsets[dst+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int src = column_indices[offset];
			if (front[src] == 1) {
				depths[dst] = depth;
				next[dst] = 1;
				break;
			}
		}
	}
}

__global__ void prepare(int m, DistT *depths, int *next) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m && depths[tid] != MYINFINITY) {
		next[tid] = 2;
	}
}

__global__ void bottom_up_tile(int m, const IndexT *row_offsets, const IndexT *column_indices, IndexT* idx_map, int *front, int *next) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		int dst = idx_map[tid];
		if (next[dst] == 0) {
			int row_begin = row_offsets[tid];
			int row_end = row_offsets[tid+1];
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int src = column_indices[offset];
				if (front[src] == 1) {
					next[dst] = 1;
					break;
				}
			}
		}
	}
}

__global__ void update(int m, int *next, DistT *depths, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		if (next[tid] == 1) {
			depths[tid] = depth;
		} else if (next[tid] == 2) {
			next[tid] = 0;
		}
	}
}

__device__ void td_expand_CTA(int m, int *row_offsets, int *column_indices, int *degrees, int *scout_count, DistT *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
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
				dst = column_indices[edge];
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					atomicAdd(scout_count, degrees[dst]);
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__device__ __forceinline__ void td_expand_warp(int m, int *row_offsets, int *column_indices, int *degrees, int *scout_count, DistT *depths, Worklist2 &in_queue, Worklist2 &out_queue, int depth) {
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
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				if(depths[dst] == MYINFINITY) {
					depths[dst] = depth;
					atomicAdd(scout_count, degrees[dst]);
					ncnt = 1;
				}
			}
			out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		}
	}
}

__global__ void top_down_base(int m, int *row_offsets, int *column_indices, int *degrees, int *scout_count, DistT *depths, Worklist2 in_queue, Worklist2 out_queue, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == MYINFINITY) && (atomicCAS(&depths[dst], MYINFINITY, depth)==MYINFINITY)) {
				assert(out_queue.push(dst));
				atomicAdd(scout_count, degrees[dst]);
			}
		}
	}
}

__global__ void top_down_lb(int m, int *row_offsets, int *column_indices, int *degrees, int *scout_count, DistT *depths, Worklist2 in_queue, Worklist2 out_queue, int depth) {
	td_expand_CTA(m, row_offsets, column_indices, degrees, scout_count, depths, in_queue, out_queue, depth);
	td_expand_warp(m, row_offsets, column_indices, degrees, scout_count, depths, in_queue, out_queue, depth);
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
		int ncnt = 0;
		int dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			if(depths[dst] == MYINFINITY) {
				depths[dst] = depth;
				atomicAdd(scout_count, degrees[dst]);
				ncnt = 1;
			}
		}
		out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

__global__ void QueueToBitmap(int num, Worklist2 queue, int *bm) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		int src;
		if (queue.pop_id(tid, src)) bm[src] = 1;
	}
}

__global__ void BitmapToQueue(int m, int *bm, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m && bm[tid]) queue.push(tid);
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *in_degree, int *h_degrees, DistT *h_depths) {
	//print_device_info(0);
#ifdef ENABLE_TILE
	segmenting(m, in_row_offsets, in_column_indices, NULL);
	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs), d_idx_map(num_subgraphs);
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
	}
#else
	int *d_in_row_offsets, *d_in_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
#endif
	DistT zero = 0;
	int *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_depths;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	int *d_scout_count;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
	int *front, *next;
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));

	int iter = 0;
	Worklist2 queue1(m), queue2(m);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int alpha = 15, beta = 18;
	int nitems = 1;
	int edges_to_check = nnz;
	int scout_count = h_degrees[source];
	
	const int nthreads = BLOCK_SIZE;
	const int nblocks = (m - 1) / nthreads + 1;
	insert<<<1, nthreads>>>(source, *in_frontier);
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		if (scout_count > edges_to_check / alpha) {
			//CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
			int awake_count, old_awake_count;
			QueueToBitmap<<<((nitems-1)/512+1), 512>>>(nitems, *in_frontier, front);
			//awake_count = thrust::reduce(thrust::device, front, front + m, 0, thrust::plus<int>());
			//printf("Transition from TD to BU: nitems=%d, awake_count=%d\n", nitems, awake_count);
			awake_count = nitems;
			do {
				++ iter;
				old_awake_count = awake_count;
#ifdef ENABLE_TILE
				prepare <<<nblocks, nthreads>>> (m, d_depths, next);
				for (int bid = 0; bid < num_subgraphs; bid ++) {
					int n_vertices = ms_of_subgraphs[bid];
					int nnz = nnzs_of_subgraphs[bid];
					int bblocks = (n_vertices - 1) / nthreads + 1;
					bottom_up_tile <<<bblocks, nthreads>>> (n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_idx_map[bid], front, next);
				}
				update <<<nblocks, nthreads>>> (m, next, d_depths, iter);
#else
				bottom_up_base <<<nblocks, nthreads>>> (m, d_in_row_offsets, d_in_column_indices, d_depths, front, next, iter);
#endif
				CudaTest("solving bottom_up failed");
				awake_count = thrust::reduce(thrust::device, next, next + m, 0, thrust::plus<int>());
				//printf("BU: (awake_count=%d) ", awake_count);
				//printf("BU: iteration=%d, num_frontier=%d\n", iter, awake_count);
				// swap the queues
				int *temp = front;
				front = next;
				next = temp;
				CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
			} while((awake_count >= old_awake_count) || (awake_count > m / beta));
			in_frontier->reset();
			BitmapToQueue<<<((m-1)/512+1), 512>>>(m, front, *in_frontier);
			scout_count = 1;
		} else {
			++ iter;
			edges_to_check -= scout_count;
			nitems = in_frontier->nitems();
			const int mblocks = (nitems - 1) / nthreads + 1;
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			if (1)
				top_down_lb <<<mblocks, nthreads>>> (m, d_out_row_offsets, d_out_column_indices, d_degrees, d_scout_count, d_depths, *in_frontier, *out_frontier, iter);
			else
				top_down_base <<<mblocks, nthreads>>> (m, d_out_row_offsets, d_out_column_indices, d_degrees, d_scout_count, d_depths, *in_frontier, *out_frontier, iter);
			CudaTest("solving top_down failed");
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			nitems = out_frontier->nitems();
			Worklist2 *tmp = in_frontier;
			in_frontier = out_frontier;
			out_frontier = tmp;
			out_frontier->reset();
			//printf("TD: (scout_count=%d) ", scout_count);
			//printf("TD: iteration=%d, num_frontier=%d\n", iter, nitems);
		}
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	//CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_scout_count));
	return;
}
