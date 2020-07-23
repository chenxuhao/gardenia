// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void push_base(int m, const uint64_t *Ap,
                          const VertexId *Aj, const ValueT *Ax,
                          const ValueT *x, ValueT *y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m) {
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		ValueT value = x[row];
		for (int offset = row_begin; offset < row_end; offset ++){
			IndexT dst = Aj[offset];
			ValueT product = Ax[offset] * value;
			atomicAdd(&y[dst], product);
		}
	}
}

__device__ void __forceinline__ expandByCta(int m, 
                                            const uint64_t *Ap, const VertexId *Aj, 
                                            const ValueT *Ax, const ValueT *x, 
                                            ValueT *y, int *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(id < m) {
		size = Ap[id+1] - Ap[id];
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
		int row_begin = Ap[sh_vertex];
		int row_end = Ap[sh_vertex+1];
		int neighbor_size = row_end - row_begin;
		ValueT value = x[sh_vertex];
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = Aj[edge];
				atomicAdd(&y[dst], value * Ax[edge]);
			}
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, 
                                             const uint64_t *Ap, const VertexId *Aj, 
                                             const ValueT *Ax, const ValueT *x, 
                                             ValueT *y, int *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	if(id < m && !processed[id]) {
		size = Ap[id+1] - Ap[id];
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
		int row_begin = Ap[winner];
		int row_end = Ap[winner+1];
		int neighbor_size = row_end - row_begin;
		ScoreT value = x[winner];
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = Aj[edge];
				atomicAdd(&y[dst], value * Ax[edge]);
			}
		}
	}
}

__global__ void push_lb(int m, uint64_t *Ap, VertexId *Aj, 
                        ValueT *Ax, ValueT *x, 
                        ValueT *y, int *processed) {
	expandByCta(m, Ap, Aj, Ax, x, y, processed);
	expandByWarp(m, Ap, Aj, Ax, x, y, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
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
		neighbor_offset = Ap[tid];
		neighbor_size = Ap[tid+1] - neighbor_offset;
		values[tx] = x[tid];
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	
	int done = 0;
	int neighbors_done = 0;
	while (total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && 
        (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			src_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int edge = gather_offsets[tx];
			int dst = Aj[edge];
			atomicAdd(&y[dst], values[src_idx[tx]] * Ax[edge]);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
	auto h_Ap = g.out_rowptr();
	auto h_Aj = g.out_colidx();	
	//print_device_info(0);
	uint64_t *d_Ap;
  VertexId *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

	int *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	push_lb <<<nblocks, nthreads>>> (m, d_Ap, d_Aj, d_Ax, d_x, d_y, d_processed);
	//push_base <<<nblocks, nthreads>>> (m, d_Ap, d_Aj, d_Ax, d_x, d_y);
	CudaTest("solving push kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [cuda_push] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
	CUDA_SAFE_CALL(cudaFree(d_processed));
}

