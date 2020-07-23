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
#define BC_VARIANT "topo_pb"
//#define REVERSE_WARP

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, int source, int *path_counts, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(id == source) {
			path_counts[id] = 1;
			depths[id] = 0;
		} else {
			depths[id] = -1;
		}
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

__global__ void forward_base(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, int depth, bool *expanded) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if(src < m && depths[src] == depth-1) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		int value = path_counts[src];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (__ldg(depths+dst) == -1) {
				visited[dst] = 1;
				atomicAdd(&path_counts[dst], value);
			}
		}
	}
}

__device__ __forceinline__ void process_edge(int dst, int value, const int *depths, int *path_counts, int *visited) {
	if (__ldg(depths+dst) == -1) {
		visited[dst] = 1;
		atomicAdd(&path_counts[dst], value);
	}
}

__global__ void update(int m, int *visited, int *depths, bool *changed, int *nitems, int *queue, int queue_len, int depth) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m && visited[tid] == 1) {
		depths[tid] = depth;
		//queue.push(tid);
		int pos = atomicAdd(nitems, 1);
		queue[queue_len + pos] = tid;
		*changed = true;
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, int depth, bool *expanded) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_src[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	if(src < m && depths[src] == depth-1) {
		size = row_offsets[src+1] - row_offsets[src];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_src[warp_id] = src;
			expanded[src] = 1;
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
			int offset = row_begin + i;
			int dst = column_indices[offset];
			if(i < neighbor_size) {
				process_edge(dst, value, depths, path_counts, visited);
			}
		}
	}
}

__global__ void forward_lb(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *depths, int *path_counts, int *visited, int depth, bool *expanded) {
	//expandByCta(m, row_offsets, column_indices, depths, path_counts, visited, depth, expanded);
	expandByWarp(m, row_offsets, column_indices, depths, path_counts, visited, depth, expanded);
	int src = blockIdx.x * blockDim.x + threadIdx.x;
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
	if(src < m && depths[src] == depth-1 && !expanded[src]) {
		expanded[src] = 1;
		neighbor_offset = row_offsets[src];
		neighbor_size = row_offsets[src+1] - neighbor_offset;
		values[threadIdx.x] = path_counts[src];
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
			srcsrc[scratch_offset + i - done] = threadIdx.x;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(threadIdx.x < total_edges) {
			int offset = gather_offsets[threadIdx.x];
			int dst = column_indices[offset];
			process_edge(dst, values[srcsrc[threadIdx.x]], depths, path_counts, visited);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void reverse_base(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num) {
		int src = frontiers[start + id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				delta_src += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		deltas[src] = delta_src;
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

__global__ void reverse_warp(int num, const IndexT *row_offsets, const IndexT *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction conditionals
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int index = warp_id; index < num; index += num_warps) {
		int src = frontiers[start + index];
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
		sdata[threadIdx.x] = sum; __syncthreads();
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
}

void BCSolver(int m, int nnz, int source, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *h_row_offsets, IndexT *h_column_indices, int *h_degrees, ScoreT *h_scores) {
	//print_device_info(0);
	int zero = 0;
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
	bool *d_changed, h_changed, *d_expanded;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_expanded, 0, m * sizeof(bool)));
	int *d_visited;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(int)));
	int *d_nitems, h_nitems = 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nitems, sizeof(int)));

	int depth = 0;
	vector<int> depth_index;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, source, d_path_counts, d_depths);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	int frontiers_len = 0;
	depth_index.push_back(0);

#ifdef REVERSE_WARP
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(reverse_warp, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
#endif
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		depth++;
		h_changed = false;
		//printf("iteration=%d, frontire_size=%d\n", depth, h_nitems);
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_nitems, &zero, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(int)));
		frontiers_len += h_nitems;
		depth_index.push_back(frontiers_len);
		forward_lb<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_depths, d_path_counts, d_visited, depth, d_expanded);
		CudaTest("solving bc_forward failed");
		update<<<nblocks, nthreads>>>(m, d_visited, d_depths, d_changed, d_nitems, d_frontiers, frontiers_len, depth);
		CudaTest("solving bc_update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&h_nitems, d_nitems, sizeof(int), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		h_nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, h_nitems);
#ifdef REVERSE_WARP
		nblocks = std::min(max_blocks, DIVIDE_INTO(h_nitems, WARPS_PER_BLOCK));
		reverse_warp<<<nblocks, nthreads>>>(h_nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
#else
		nblocks = (h_nitems - 1) / nthreads + 1;
		reverse_lb<<<nblocks, nthreads>>>(h_nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
#endif
		CudaTest("solving kernel2 failed");
	}
	
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	//printf("\nStart calculating the maximum score\n");
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	//h_max_score = *max_element(h_scores, h_scores+m);
	//for (int n = 0; n < m; n ++) h_scores[n] = h_scores[n] / h_max_score;
	//std::cout << "max_score = " << h_max_score << "\n";
	//printf("\nStart normalizing scores\n");
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

