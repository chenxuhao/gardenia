// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define GPU_SEGMENTING
#include "segmenting.h"
//#define ENABLE_WARP
#define SPMV_VARIANT "partition"

typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

template<typename T>
__global__ void initialize(int m, T *sums) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) sums[id] = 0;
}

template<typename T>
__device__ __inline__ T ld_glb_cs(const T *addr) {
	T return_value;
	//asm("ld.cs.global.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	asm("ld.cs.global.f32 %0, [%1];" : "=f"(return_value) : "l"(addr));
	return return_value;
}

template<typename T>
__device__ __inline__ void st_glb_cs(T value, T *addr) {
	asm("st.cs.global.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__device__ __forceinline__ void expandByCta(int m, const IndexT *Ap, const IndexT *Aj, const ValueT *Ax, const ValueT * x, ValueT *partial_sums, int *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ int owner;
	__shared__ int row;
	owner = -1;
	int size = 0;
	if(id < m) size = Ap[id+1] - Ap[id];
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			row = id;
			processed[id] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		ValueT sum = 0;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				//sum += Ax[offset] * x[Aj[offset]];
				sum += Ax[offset] * __ldg(x+Aj[offset]);
			}
		}
		ValueT block_sum = BlockReduce(temp_storage).Sum(sum);
		if(threadIdx.x == 0) partial_sums[row] = block_sum;
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __forceinline__ void expandByWarp(int m, const IndexT *Ap, const IndexT *Aj, const ValueT *Ax, const ValueT * x, ValueT *partial_sums, int *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];
	owner[warp_id] = -1;
	int size = 0;
	if(id < m && !processed[id]) {
		size = Ap[id+1] - Ap[id];
	}
	//while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
	while(__any(size) >= WARP_SIZE) {
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
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		ScoreT sum = 0;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int offset = row_begin + i;
			if(i < neighbor_size) {
				//sum += Ax[offset] * x[Aj[offset]];
				sum += Ax[offset] * __ldg(x+Aj[offset]);
			}
		}
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(lane_id == 0) partial_sums[winner] += sdata[threadIdx.x];
	}
}

__global__ void spmv_base(int m, const IndexT * Ap, const IndexT * Aj, const ValueT * Ax, const ValueT * x, ValueT * partial_sums, int *processed) {
	expandByCta(m, Ap, Aj, Ax, x, partial_sums, processed);
	expandByWarp(m, Ap, Aj, Ax, x, partial_sums, processed);
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m && !processed[row]) {
		int row_begin = Ap[row];
		int row_end = Ap[row+1];
		ValueT sum = 0;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int col = Aj[offset];
			//int col = __ldg(Aj+offset);
			//sum += Ax[offset] * x[Aj[offset]];
			//sum += Ax[offset] * __ldg(x+Aj[offset]);
			sum += ld_glb_cs<ValueT>(Ax+offset) * __ldg(x+col);
		}
		//partial_sums[row] = sum;
		st_glb_cs<ValueT>(sum, partial_sums+row);
	}
}

__global__ void spmv_warp(int m, const IndexT * Ap, const IndexT * Aj, const ValueT * Ax, const ValueT * x, ValueT *partial_sums) {
	__shared__ ValueT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < m; row += num_warps) {
		if (thread_lane < 2)
			ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];
		ValueT sum = 0;
		for (int offset = row_start + thread_lane; offset < row_end; offset += WARP_SIZE)
			sum += Ax[offset] * __ldg(x+Aj[offset]);

		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if (thread_lane == 0) partial_sums[row] += sdata[threadIdx.x];
	}
}

__global__ void merge_cta(int m, int num_subgraphs, IndexT** range_indices, IndexT** idx_map, ValueT** partial_sums, ValueT *y) {
	int rid = blockIdx.x;
	int tx  = threadIdx.x;
	__shared__ ValueT sdata[RANGE_WIDTH];
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		sdata[tx + i] = 0;
	}
	__syncthreads();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int start = range_indices[bid][rid];
		int end = range_indices[bid][rid+1];
		int size = end - start;
		int num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for (int i = tx; i < num; i += blockDim.x) {
			int lid = start + i;
			if (i < size) {
				int gid = idx_map[bid][lid];
				ValueT local_sum = partial_sums[bid][lid];
				sdata[gid%RANGE_WIDTH] += local_sum;
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < RANGE_WIDTH; i += BLOCK_SIZE) {
		int local_id = tx + i;
		int global_id = rid * RANGE_WIDTH + local_id;
		if (global_id < m)
			y[global_id] += sdata[local_id];
	}
}

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *h_Ap, IndexT *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *h_y, int *degrees) { 
	//print_device_info(0);
	segmenting(m, h_Ap, h_Aj, h_Ax);
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);
	
	ValueT *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (m - 1) / RANGE_WIDTH + 1;
	vector<IndexT *> d_Ap_blocked(num_subgraphs), d_Aj_blocked(num_subgraphs);
	vector<ValueT *> d_Ax_blocked(num_subgraphs);
	IndexT ** d_range_indices = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	IndexT ** d_idx_map = (IndexT**)malloc(num_subgraphs * sizeof(IndexT*));
	ValueT ** d_partial_sums = (ValueT**)malloc(num_subgraphs * sizeof(ValueT*));

	printf("copy host data to device\n");
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_range_indices[bid], (num_ranges+1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_partial_sums[bid], ms_of_subgraphs[bid] * sizeof(ValueT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ap_blocked[bid], rowptr_blocked[bid], (ms_of_subgraphs[bid] + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Aj_blocked[bid], colidx_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_Ax_blocked[bid], values_blocked[bid], nnzs_of_subgraphs[bid] * sizeof(ValueT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_idx_map[bid], idx_map[bid], ms_of_subgraphs[bid] * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_range_indices[bid], range_indices[bid], (num_ranges+1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	}

	printf("copy host pointers to device\n");
	IndexT ** d_range_indices_ptr, **d_idx_map_ptr;
	ValueT ** d_partial_sums_ptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_range_indices_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_idx_map_ptr, num_subgraphs * sizeof(IndexT*)));
	CUDA_SAFE_CALL(cudaMalloc(&d_partial_sums_ptr, num_subgraphs * sizeof(ValueT*)));
	CUDA_SAFE_CALL(cudaMemcpy(d_range_indices_ptr, d_range_indices, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_idx_map_ptr, d_idx_map, num_subgraphs * sizeof(IndexT*), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_partial_sums_ptr, d_partial_sums, num_subgraphs * sizeof(ValueT*), cudaMemcpyHostToDevice));
	int *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));

	const int nthreads = BLOCK_SIZE;
	int mblocks = (m - 1) / nthreads + 1;
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		int msub = ms_of_subgraphs[bid];
		mblocks = (msub - 1) / nthreads + 1;
		initialize<ValueT> <<<mblocks, nthreads>>> (msub, d_partial_sums[bid]);
	}
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	int max_blocks_per_SM = maximum_residency(spmv_warp, nthreads, 0);
	int max_blocks = max_blocks_per_SM * nSM;
	int nblocks = std::min(max_blocks, DIVIDE_INTO(m, WARPS_PER_BLOCK));
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	for (int bid = 0; bid < num_subgraphs; bid ++) {
		//Timer tt;
		//tt.Start();
		int msub = ms_of_subgraphs[bid];
		int nnz = nnzs_of_subgraphs[bid];
#ifdef ENABLE_WARP
		nblocks = std::min(max_blocks, DIVIDE_INTO(msub, WARPS_PER_BLOCK));
		spmv_warp<<<nblocks, nthreads>>>(msub, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_partial_sums[bid]);
#else
		CUDA_SAFE_CALL(cudaMemset(d_processed, 0, m * sizeof(int)));
		int bblocks = (msub - 1) / nthreads + 1;
		spmv_base<<<bblocks, nthreads>>>(msub, d_Ap_blocked[bid], d_Aj_blocked[bid], d_Ax_blocked[bid], d_x, d_partial_sums[bid], d_processed);
#endif
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
		//tt.Stop();
		//printf("\truntime subgraph[%d] = %f ms.\n", bid, tt.Millisecs());
	}
	CudaTest("solving spmv kernel failed");
	merge_cta <<<num_ranges, nthreads>>>(m, num_subgraphs, d_range_indices_ptr, d_idx_map_ptr, d_partial_sums_ptr, d_y);
	CudaTest("solving merge kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [%s] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", SPMV_VARIANT, time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

