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
#define BC_VARIANT "linear_lb"
//#define REVERSE_WARP

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) depths[id] = -1;
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(const IndexT *row_offsets, 
                           const IndexT *column_indices, 
                           int *path_counts, int *depths, 
                           int depth, Worklist2 in_queue, 
                           Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
				assert(out_queue.push(dst));
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, const IndexT *row_offsets, 
                           const IndexT *column_indices, 
                           int start, const IndexT *frontiers, 
                           ScoreT *scores, const int *path_counts, 
                           int *depths, int depth, ScoreT *deltas) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		int src = frontiers[start + tid];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (depths[dst] == depth + 1) {
				deltas[src] += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
			}
		}
		scores[src] += deltas[src];
	}
}

__device__ __forceinline__ void process_edge(int value, int depth, int offset, 
                                             const IndexT *column_indices, 
                                             int *path_counts, int *depths, 
                                             Worklist2 &out_queue) {
	int dst = column_indices[offset];
	if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
		assert(out_queue.push(dst));
	}
	if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
}

__device__ __forceinline__ void expandByCta(const uint64_t *row_offsets, 
                                            const IndexT *column_indices, 
                                            int *path_counts, int *depths, 
                                            Worklist2 &in_queue, 
                                            Worklist2 &out_queue, int depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_src;
	owner = -1;
	int size = 0;
	int src;
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
		int value = path_counts[sh_src];
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int dst = 0;
			int ncnt = 0;
			int offset = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[offset];
				if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1))
					ncnt = 1;
				if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
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

__device__ __forceinline__ void expandByWarp(const uint64_t* row_offsets, 
                                             const IndexT *column_indices, 
                                             int *path_counts, int *depths, 
                                             Worklist2 &in_queue, 
                                             Worklist2 &out_queue, int depth) {
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
		int value = path_counts[winner];
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				process_edge(value, depth, edge, column_indices, path_counts, depths, out_queue);
			}
		}
	}
}

__global__ void forward_lb(const uint64_t* row_offsets, 
                           const IndexT *column_indices, 
                           int *path_counts, int *depths, 
                           int depth, Worklist2 in_queue, 
                           Worklist2 out_queue) {
	expandByCta(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
	expandByWarp(row_offsets, column_indices, path_counts, depths, in_queue, out_queue, depth);
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
			int edge = gather_offsets[tx];
			process_edge(values[srcsrc[tx]], depth, edge, column_indices, path_counts, depths, out_queue);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void bc_reverse_warp(int num, const uint64_t* row_offsets, 
                                const IndexT *column_indices, int start, 
                                const IndexT *frontiers, ScoreT *scores, 
                                const int *path_counts, int *depths, 
                                int depth, ScoreT *deltas) {
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
					static_cast<ScoreT>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
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
}

__device__ __forceinline__ void reverse_expand_cta(int num, 
                                                   const uint64_t* row_offsets, 
                                                   const IndexT *column_indices, 
                                                   int start, IndexT *frontiers, 
                                                   ScoreT *scores, const int *path_counts, 
                                                   int *depths, int depth, ScoreT *deltas) {
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

__device__ __forceinline__ void reverse_expand_warp(int num, 
                                                    const uint64_t* row_offsets, 
                                                    const IndexT *column_indices, 
                                                    int start, IndexT *frontiers, 
                                                    ScoreT *scores, const int *path_counts, 
                                                    int *depths, int depth, ScoreT *deltas) {
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

__global__ void reverse_lb(int num, const uint64_t* row_offsets, 
                           const IndexT *column_indices, int start, 
                           IndexT *frontiers, ScoreT *scores, 
                           const int *path_counts, int *depths, 
                           int depth, ScoreT *deltas) {
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
				//ScoreT value = static_cast<ScoreT>(path_counts[srcs[tx]]) / 
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

void BCSolver(Graph &g, int source, ScoreT *h_scores) {
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

#ifdef REVERSE_WARP
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  const int nSM = deviceProp.multiProcessorCount;
  const int max_blocks_per_SM = maximum_residency(bc_reverse, nthreads, 0);
  const int max_blocks = max_blocks_per_SM * nSM;
#endif
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
    forward_lb<<<nblocks, nthreads>>>(d_row_offsets, d_column_indices, d_path_counts, d_depths, depth, *inwl, *outwl);
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
    nblocks = (nitems - 1) / nthreads + 1;
    //printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
#ifdef REVERSE_WARP
    nblocks = std::min(max_blocks, DIVIDE_INTO(nitems, WARPS_PER_BLOCK));
    bc_reverse_warp<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
#else
    reverse_lb<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
#endif
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
  printf("\tmax_score = %.6f.\n", h_max_score);
  printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_path_counts));
  CUDA_SAFE_CALL(cudaFree(d_depths));
  CUDA_SAFE_CALL(cudaFree(d_deltas));
  CUDA_SAFE_CALL(cudaFree(d_frontiers));
  CUDA_SAFE_CALL(cudaFree(d_row_offsets));
  CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

