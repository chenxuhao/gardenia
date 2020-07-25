// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "partition"
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define GPU_SEGMENTING
#include "segmenting.h"
//#define ENABLE_WARP
#define ENABLE_CTA

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
}

template<typename T>
__device__ __inline__ T ld_glb_cs(const T *addr) {
	T return_value;
	asm("ld.cs.global.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	return return_value;
}

template<typename T>
__device__ __inline__ void st_glb_cs(T value, T *addr) {
	asm("st.cs.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__device__ __forceinline__ void expandByCta(int m, const IndexT *row_offsets, const IndexT *column_indices, ScoreT *sums, const ScoreT *outgoing_contrib, bool *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	int dst = id;
	if(dst < m) size = row_offsets[dst+1] - row_offsets[dst];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = dst;
			processed[dst] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		ScoreT sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				//int src = column_indices[edge];
				int src = __ldg(column_indices+edge);
				//sum += outgoing_contrib[src];
				sum += __ldg(outgoing_contrib+src);
			}
		}
		ScoreT block_sum = BlockReduce(temp_storage).Sum(sum);
		if(threadIdx.x == 0) sums[sh_vertex] = block_sum;
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *row_offsets, const IndexT *column_indices, ScoreT *sums, const ScoreT *outgoing_contrib, bool *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];
	owner[warp_id] = -1;
	int size = 0;
	int dst = id;
	if(dst < m && !processed[dst]) {
		size = row_offsets[dst+1] - row_offsets[dst];
	}
	//while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = dst;
			processed[dst] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		ScoreT sum = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				//int src = column_indices[edge];
				int src = __ldg(column_indices+edge);
				//sum += outgoing_contrib[src];
				sum += __ldg(outgoing_contrib+src);
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) sums[winner] += sdata[threadIdx.x];
	}
}

__global__ void pull_base(int m, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, ScoreT *partial_sums, const __restrict__ ScoreT *outgoing_contrib, bool *processed) {
	expandByCta(m, row_offsets, column_indices, partial_sums, outgoing_contrib, processed);
	expandByWarp(m, row_offsets, column_indices, partial_sums, outgoing_contrib, processed);
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	//if (dst < m) {
	if (dst < m && !processed[dst]) {
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		//IndexT row_begin = __ldg(row_offsets+dst);
		//IndexT row_end = __ldg(row_offsets+dst+1);
		//IndexT row_begin = ld_glb_cs<IndexT>(row_offsets+dst);
		//IndexT row_end = ld_glb_cs<IndexT>(row_offsets+dst+1);
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			//IndexT src = column_indices[offset];
			IndexT src = __ldg(column_indices+offset);
			//IndexT src = ld_glb_cs<IndexT>(column_indices+offset);
			//incoming_total += outgoing_contrib[src];
			incoming_total += __ldg(outgoing_contrib+src);
		}
		//partial_sums[dst] = incoming_total;
		st_glb_cs<ScoreT>(incoming_total, partial_sums+dst);
	}
}

__global__ void pull_lb(int m, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, ScoreT *partial_sums, const __restrict__ ScoreT *outgoing_contrib, bool *processed) {
	expandByCta(m, row_offsets, column_indices, partial_sums, outgoing_contrib, processed);
	expandByWarp(m, row_offsets, column_indices, partial_sums, outgoing_contrib, processed);
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int dst_idx[BLOCK_SIZE];
	__shared__ ScoreT sh_total[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	dst_idx[tx] = 0;
	sh_total[tx] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (dst < m && !processed[dst]) {
		neighbor_offset = row_offsets[dst];
		neighbor_size = row_offsets[dst+1] - neighbor_offset;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while (total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			dst_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			int src = column_indices[offset];
			atomicAdd(&sh_total[dst_idx[tx]], outgoing_contrib[src]); 
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	__syncthreads();
	if (dst < m && !processed[dst])
		partial_sums[dst] = sh_total[tx];
		//st_glb_cs<ScoreT>(sh_total, partial_sums+dst);
}

__global__ void pull_warp_kernel(int m, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, ScoreT *partial_sums, const __restrict__ ScoreT *outgoing_contrib) {
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int dst = warp_id; dst < m; dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		ScoreT sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int src = column_indices[offset];
			sum += outgoing_contrib[src];
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if (thread_lane == 0) partial_sums[dst] += sdata[threadIdx.x];
	}
}

template <int VECTORS_PER_BLOCK, int THREADS_PER_VECTOR>
__global__ void pull_vector_kernel(int m, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, ScoreT *partial_sums, const __restrict__ ScoreT *outgoing_contrib) {
	__shared__ ScoreT sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]; // padded to avoid reduction ifs
	__shared__ int ptrs[VECTORS_PER_BLOCK][2];

	const int thread_id	  = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR-1);   // thread index within the vector
	const int vector_id   = thread_id   / THREADS_PER_VECTOR;       // global vector index
	const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;       // vector index within the CTA
	const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;          // total number of active vectors

	for(int dst = vector_id; dst < m; dst += num_vectors) {
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_start = ptrs[vector_lane][0];                   //same as: row_start = row_offsets[row];
		const int row_end   = ptrs[vector_lane][1];                   //same as: row_end   = row_offsets[row+1];

		// compute local sum
		ScoreT sum = 0;
		for(int offset = row_start + thread_lane; offset < row_end; offset += THREADS_PER_VECTOR) {
			//int src = column_indices[offset];
			int src = __ldg(column_indices+offset);
			//sum += outgoing_contrib[src];
			sum += __ldg(outgoing_contrib+src);
		}

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum; __syncthreads();
		if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes vector result
		//if (thread_lane == 0) partial_sums[dst] += sdata[threadIdx.x];
		if (thread_lane == 0) st_glb_cs(sdata[threadIdx.x], partial_sums+dst);
	}
}

__global__ void merge(int num_ranges, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ScoreT** partial_sums, ScoreT *sums) {
	int rid = blockIdx.x * blockDim.x + threadIdx.x;
	if(rid < num_ranges) {
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			int start = range_indices[bid][rid];
			int end = range_indices[bid][rid+1];
			for (int lid = start; lid < end; lid ++) {
				//int gid = idx_map[bid][lid];
				int gid = __ldg(idx_map[bid]+lid);
				ScoreT local_sum = partial_sums[bid][lid];
				sums[gid] += local_sum;
			}
		}
	}
}

__global__ void merge_cta(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ScoreT** partial_sums, ScoreT *sums) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ ScoreT sdata[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		sdata[tx + i] = 0;
	}
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		//int start = __ldg(range_indices[bid]+rid);
		//int end = __ldg(range_indices[bid]+rid+1);
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				//int gid = __ldg(idx_map[bid]+lid);
				ScoreT local_sum = partial_sums[bid][lid];
				sdata[gid%RANGE_WIDTH] += local_sum;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m)
			sums[global_id] = sdata[local_id];
	}
}

__global__ void l1norm(int m, ScoreT *scores, ScoreT *sums, float *diff, ScoreT base_score) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0;
	if(u < m) {
		ScoreT new_score = base_score + kDamp * sums[u];
		local_diff += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

template <int THREADS_PER_VECTOR>
void pull_vector(int m, int nSM, const __restrict__ IndexT *row_offsets, const __restrict__ IndexT *column_indices, ScoreT *partial_sums, ScoreT *outgoing_contrib) {
	const int VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	const int max_blocks_per_SM = maximum_residency(pull_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(max_blocks, DIVIDE_INTO(m, VECTORS_PER_BLOCK));
	pull_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<nblocks, BLOCK_SIZE>>>(m, row_offsets, column_indices, partial_sums, outgoing_contrib);
	CudaTest("solving failed");
}

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	//print_device_info(0);
	segmenting(m, in_row_offsets, in_column_indices, NULL);

	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores, *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (m - 1) / RANGE_WIDTH + 1;
	vector<IndexT *> d_row_offsets_blocked(num_subgraphs), d_column_indices_blocked(num_subgraphs);
	IndexT ** d_range_indices = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	ScoreT ** d_partial_sums = (ScoreT**)malloc(num_subgraphs * sizeof(ScoreT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_range_indices[bid], (num_ranges+1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_sums[bid], ms_of_subgraphs[bid] * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_range_indices[bid], range_indices[bid], (num_ranges+1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(d_partial_sums[bid], 0, ms_of_subgraphs[bid] * sizeof(ScoreT)));
	}

	printf("copy host pointers to device\n");
	IndexT ** d_range_indices_ptr, **d_idx_map_ptr;
	ScoreT ** d_partial_sums_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_range_indices_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_idx_map_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_sums_ptr, num_subgraphs * sizeof(ScoreT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_range_indices_ptr, d_range_indices, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_idx_map_ptr, d_idx_map, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_sums_ptr, d_partial_sums, num_subgraphs * sizeof(ScoreT*), cudaMemcpyHostToDevice));
	bool *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(bool)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int nblocks = (m - 1) / nthreads + 1;
#ifndef ENABLE_CTA
	int mblocks = (num_ranges - 1) / nthreads + 1;
#endif
#ifdef ENABLE_WARP
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
	const int max_blocks_per_SM = maximum_residency(pull_warp_kernel, nthreads, 0);
	const int max_blocks = max_blocks_per_SM * nSM;
#endif
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
		for (int bid = 0; bid < num_subgraphs; bid ++) {
			//Timer tt;
			//tt.Start();
			int n_vertices = ms_of_subgraphs[bid];
			int nnz = nnzs_of_subgraphs[bid];
			int bblocks = (n_vertices - 1) / nthreads + 1;
			CUDA_SAFE_CALL(cudaMemset(d_processed, 0, n_vertices * sizeof(bool)));
#ifndef ENABLE_WARP
			pull_base <<<bblocks, nthreads>>>(n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib, d_processed);
#else
			initialize <<<bblocks, nthreads>>> (n_vertices, d_partial_sums[bid]);
			///*
			int wblocks = std::min(max_blocks, DIVIDE_INTO(n_vertices, WARPS_PER_BLOCK));
			pull_warp_kernel <<<wblocks, nthreads>>> (n_vertices, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib); 
			//*/
			/*
			int nnz_per_row = nnz / n_vertices;
			if (nnz_per_row <=  2)
				pull_vector<2>(n_vertices, nSM, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib);
			else if (nnz_per_row <=  4)
				pull_vector<4>(n_vertices, nSM, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib);
			else if (nnz_per_row <=  8)
				pull_vector<8>(n_vertices, nSM, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib);
			else if (nnz_per_row <= 16)
				pull_vector<16>(n_vertices, nSM, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib);
			else
				pull_vector<32>(n_vertices, nSM, d_row_offsets_blocked[bid], d_column_indices_blocked[bid], d_partial_sums[bid], d_contrib);
			//*/
#endif
			//CUDA_SAFE_CALL(cudaDeviceSynchronize());
			//tt.Stop();
			//if(iter == 1) printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
		}
		CudaTest("solving kernel pull_step failed");
#ifndef ENABLE_CTA
		merge <<<mblocks, nthreads>>>(num_ranges, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_sums_ptr, d_sums);
#else
		merge_cta <<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_sums_ptr, d_sums);
#endif
		CudaTest("solving kernel merge failed");
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
		CudaTest("solving kernel l1norm failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));

	//CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	//CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
