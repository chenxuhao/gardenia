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
#define BC_VARIANT "hybrid_lb"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

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

__global__ void forward_lb(const IndexT *row_offsets, const IndexT *column_indices, const int *degrees, int *depths, int *path_counts, int *scout_count, int depth, Worklist2 in_queue, Worklist2 out_queue) {
	//expandByCta(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	//expandByWarp(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
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

__global__ void forward_push_lb(const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, Worklist2 in_queue) {
	//expandByCta(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	//expandByWarp(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
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

__global__ void forward_pull(int m, const IndexT *row_offsets, const IndexT *column_indices, int *depths, int *path_counts, int depth) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if(dst < m && depths[dst] == -1) { // not visited
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		int incoming_total = 0;
		bool is_next = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT src = column_indices[offset];
			if(depths[src] == depth-1) {
				is_next = 1;
				incoming_total += path_counts[src];
			}
		}
		if (is_next) depths[dst] = depth;
		path_counts[dst] = incoming_total;
	}
}

__global__ void update_pull(const int *depths, int *next, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(depths[tid] == depth)
		next[tid] = 1;
}

__global__ void update_push(const int *status, const int *degrees, int *depths, Worklist2 queue, int *scout_count, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(status[tid] == 1) {
	//if(depths[tid] == -1 && status[tid] == 1) {
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
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (depths[dst] == depth + 1) {
				deltas[src] += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		scores[src] += deltas[src];
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

void BCSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *h_degrees, ScoreT *h_scores) {
	//print_device_info(0);
	int *d_in_row_offsets, *d_in_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
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
			//QueueToBitmap<<<((nitems-1)/512+1), 512>>>(nitems, *inwl, front);
			awake_count = nitems;
			do {
				++ depth;
				int nblocks = (awake_count - 1) / nthreads + 1;
				push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
				frontiers_len += awake_count;
				depth_index.push_back(frontiers_len);
				old_awake_count = awake_count;
				//printf("BU: iteration=%d, num_frontier=%d\n", depth, awake_count);
				forward_pull <<<mblocks, nthreads>>> (m, d_in_row_offsets, d_in_column_indices, d_depths, d_path_counts, depth);
				CudaTest("solving forward failed");
				CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
				update_pull<<<mblocks, nthreads>>>(d_depths, next, depth);
				awake_count = thrust::reduce(thrust::device, next, next + m, 0, thrust::plus<int>());
				//int *temp = front;
				//front = next;
				//next = temp;
				inwl->reset();
				BitmapToQueue<<<((m-1)/512+1), 512>>>(m, next, *inwl);
			} while((awake_count >= old_awake_count) || (awake_count > m / beta));
			inwl->reset();
			BitmapToQueue<<<((m-1)/512+1), 512>>>(m, next, *inwl);
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
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			if (nitems <256) {
				forward_lb<<<nblocks, nthreads>>>(d_out_row_offsets, d_out_column_indices, d_degrees, d_depths, d_path_counts, d_scout_count, depth, *inwl, *outwl);
				CudaTest("solving kernel forward failed");
			} else {
				forward_push_lb<<<nblocks, nthreads>>>(d_out_row_offsets, d_out_column_indices, d_depths, d_path_counts, d_status, *inwl);
				CudaTest("solving kernel forward failed");
				update_push<<<mblocks, nthreads>>>(d_status, d_degrees, d_depths, *outwl, d_scout_count, depth);
				CudaTest("solving kernel update failed");
			}
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
		int nblocks = (nitems - 1) / nthreads + 1;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
		reverse_lb<<<nblocks, nthreads>>>(nitems, d_out_row_offsets, d_out_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving kernel reverse failed");
	}
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	nthreads = 512;
	bc_normalize<<<mblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
}

