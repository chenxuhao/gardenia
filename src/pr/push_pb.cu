// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define GTX1080_PB
#include "prop_blocking.h"
#define ENABLE_LB
#define PR_VARIANT "push_pb"

typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

template<typename T>
__device__ __inline__ void st_glb_cs(T value, T *addr) {
	asm("st.cs.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__global__ void initialize(int m, ScoreT *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

__global__ void contrib(int m, ScoreT *scores, int *degrees, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degrees[u];
}

__device__ __forceinline__ void expandByCta(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *pos, const ScoreT *scores, ScoreT **contri_bins, int *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(id < m) {
		size = row_offsets[id+1] - row_offsets[id];
	}
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = id;
			processed[id] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = row_offsets[sh_vertex];
		int row_end = row_offsets[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		ScoreT value = scores[sh_vertex] / (ScoreT)neighbor_size;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				//int dst = column_indices[offset];
				int dst = __ldg(column_indices+offset);
				int dest_bin = dst >> BITS;
				st_glb_cs<ScoreT>(value, contri_bins[dest_bin]+pos[offset]);
			}
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const int *row_offsets, const int *column_indices, const int *pos, const ScoreT *scores, ScoreT **contri_bins, int *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	if(id < m && !processed[id]) {
		size = row_offsets[id+1] - row_offsets[id];
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = id;
			processed[id] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = row_offsets[winner];
		int row_end = row_offsets[winner+1];
		int neighbor_size = row_end - row_begin;
		ScoreT value = scores[winner] / (ScoreT)neighbor_size;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				//int dst = column_indices[offset];
				int dst = __ldg(column_indices+offset);
				int dest_bin = dst >> BITS;
				st_glb_cs<ScoreT>(value, contri_bins[dest_bin]+pos[offset]);
			}
		}
	}
}

__global__ void binning_lb(int m, int *row_offsets, int *column_indices, int *pos, ScoreT *scores, ScoreT **contri_bins, int *processed) {
	expandByCta(m, row_offsets, column_indices, pos, scores, contri_bins, processed);
	expandByWarp(m, row_offsets, column_indices, pos, scores, contri_bins, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int src = tid;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int src_idx[BLOCK_SIZE];
	__shared__ ScoreT values[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	src_idx[tx] = 0;
	values[tx] = 0;
	__syncthreads();

	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (tid < m && !processed[tid]) {
		neighbor_offset = row_offsets[tid];
		neighbor_size = row_offsets[tid+1] - neighbor_offset;
		values[tx] = scores[src] / (ScoreT)neighbor_size;
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
			src_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int offset = gather_offsets[tx];
			//int dst = column_indices[offset];
			int dst = __ldg(column_indices+offset);
			int dest_bin = dst >> BITS;
			st_glb_cs<ScoreT>(values[src_idx[tx]], contri_bins[dest_bin]+pos[offset]);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

__global__ void binning(int m, int *row_offsets, int *column_indices, int *pos, ScoreT *scores, ScoreT **contri_bins, int *processed) {
	//expandByCta(m, row_offsets, column_indices, pos, scores, contri_bins, processed);
	//expandByWarp(m, row_offsets, column_indices, pos, scores, contri_bins, processed);
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && !processed[src]) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		int degree = row_end - row_begin;
		ScoreT value = scores[src] / (ScoreT)degree;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			//int dst = column_indices[offset];
			int dst = __ldg(column_indices+offset);
			int dest_bin = dst >> BITS;
			//contri_bins[dest_bin][pos[offset]] = value;
			st_glb_cs<ScoreT>(value, contri_bins[dest_bin]+pos[offset]);
		}
	}
}

__global__ void accumulate_shm(int m, int *sizes, IndexT **vertex_bins, ScoreT **contri_bins, ScoreT *sums) {
	int tx  = threadIdx.x;
	int bid = blockIdx.x;
	__shared__ ScoreT sdata[BIN_WIDTH];
	for (int i = 0; i < BIN_WIDTH; i += BLOCK_SIZE)
		sdata[tx+i] = 0;
	__syncthreads();
	int size = sizes[bid];
	int max = ((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
	for (int k = 0; k < max; k += BLOCK_SIZE) {
		int id = tx + k;
		if (id < size) {
			ScoreT c = contri_bins[bid][id];
			IndexT v = vertex_bins[bid][id];
			atomicAdd(&sdata[v%BIN_WIDTH], c);
		}
	}
	__syncthreads();
	for (int i = 0; i < BIN_WIDTH; i += BLOCK_SIZE) {
		int start = bid << BITS;
		int lid = tx + i;
		int gid = start + lid;
		if(gid < m) sums[gid] = sdata[lid];
	}
}

__global__ void accumulate_base(int m, int *sizes, IndexT **vertex_bins, ScoreT **contri_bins, ScoreT *sums) {
	int tx  = threadIdx.x;
	int bid = blockIdx.x;
	int size = sizes[bid];
	int max = ((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
	for (int k = 0; k < max; k += BLOCK_SIZE) {
		int id = tx + k;
		if (id < size) {
			ScoreT c = contri_bins[bid][id];
			IndexT v = vertex_bins[bid][id];
			atomicAdd(&sums[v], c);
		}
	}
}

__global__ void accumulate(int m, int num, int *sizes, IndexT **vertex_bins, ScoreT **contri_bins, ScoreT *sums) {
	int tx  = threadIdx.x;
	int bx = blockIdx.x;
	__shared__ ScoreT sdata[BIN_WIDTH];
	int start = bx * num;
	int end = start + num;

	for (int bid = start; bid < end; bid ++) {
		for (int i = 0; i < BIN_WIDTH; i += BLOCK_SIZE)
			sdata[tx+i] = 0;
		__syncthreads();
		int size = sizes[bid];
		int max = ((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
		for (int k = 0; k < max; k += BLOCK_SIZE) {
			int id = tx + k;
			if (id < size) {
				ScoreT c = contri_bins[bid][id];
				IndexT v = vertex_bins[bid][id];
				atomicAdd(&sdata[v%BIN_WIDTH], c);
			}
		}
		__syncthreads();
		for (int i = 0; i < BIN_WIDTH; i += BLOCK_SIZE) {
			int start = bid << BITS;
			int lid = tx + i;
			int gid = start + lid;
			if(gid < m) sums[gid] = sdata[lid];
		}
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

void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degrees, ScoreT *h_scores) {
	preprocessing(m, nnz, h_row_offsets, h_column_indices);
	int num_bins = (m-1) / BIN_WIDTH + 1;
	printf("bin width: %d, number of bins: %d\n", BIN_WIDTH, num_bins);

	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));

	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	ScoreT *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	int *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));

	IndexT *d_pos, *d_sizes;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pos, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sizes, num_bins * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_pos, pos.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_sizes, sizes.data(), num_bins * sizeof(int), cudaMemcpyHostToDevice));

	IndexT ** d_vertex_bins = (IndexT**)malloc(num_bins * sizeof(IndexT*));
	ScoreT ** d_contri_bins = (ScoreT**)malloc(num_bins * sizeof(ScoreT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_bins; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_vertex_bins[bid], sizes[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_contri_bins[bid], sizes[bid] * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_vertex_bins[bid], vertex_bins[bid].data(), sizes[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_contri_bins[bid], value_bins[bid].data(), sizes[bid] * sizeof(ScoreT), cudaMemcpyHostToDevice));
	}

	printf("copy host pointers to device\n");
	IndexT ** d_vertex_bins_ptr;
	ScoreT ** d_contri_bins_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_vertex_bins_ptr, num_bins * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_contri_bins_ptr, num_bins * sizeof(ScoreT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_vertex_bins_ptr, d_vertex_bins, num_bins * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_contri_bins_ptr, d_contri_bins, num_bins * sizeof(ScoreT*), cudaMemcpyHostToDevice));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	const ScoreT base_score = (1.0f - kDamp) / m;
	initialize <<<nblocks, nthreads>>> (m, d_sums);
	CudaTest("init failed");
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
#ifdef FUSED
    cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int max_blocks = maximum_residency(accumulate, nthreads, 0);
	int nSM = deviceProp.multiProcessorCount;
	int mblocks = nSM * max_blocks;
	int num = (num_bins - 1) / mblocks + 1;
	printf("%d SMs, maximum %d blocks/CTA, total %d ...\n", nSM, max_blocks, mblocks);
#endif

	Timer t;
	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));
		contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
		binning <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_pos, d_scores, d_contri_bins_ptr, d_processed);
		//binning_lb <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_pos, d_scores, d_contri_bins_ptr, d_processed);
		CudaTest("solving kernel binning failed");
#ifdef FUSED
		accumulate <<<mblocks, nthreads>>> (m, num, d_sizes, d_vertex_bins_ptr, d_contri_bins_ptr, d_sums);
#else
		//accumulate_base <<<num_bins, nthreads>>> (m, d_sizes, d_vertex_bins_ptr, d_contri_bins_ptr, d_sums);
		accumulate_shm <<<num_bins, nthreads>>> (m, d_sizes, d_vertex_bins_ptr, d_contri_bins_ptr, d_sums);
#endif
		CudaTest("solving kernel accumulate failed");
		l1norm <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_diff, base_score);
		CudaTest("solving kernel l1norm failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		//printf("iteration=%d, diff=%f\n", iter, h_diff);
		printf(" %2d    %lf\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
