#define SSSP_VARIANT "load-balance"
#include "worklistc.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define BLKSIZE 128
typedef unsigned DistT;

__global__ void initialize(int m, DistT *dist) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__device__ __forceinline__ unsigned process_edge(int src, int edge, int *column_indices, W_TYPE *weight, DistT *dist, Worklist2 &outwl) {
	int dst = column_indices[edge];
	DistT wt = (DistT)weight[edge];
	DistT altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
		//atomicMin((unsigned *)&dist[dst], altdist);
		atomicMin(&dist[dst], altdist);
		outwl.push(dst);
	}
}
typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__device__ void expandByCta(int m, int *row_offsets, int *column_indices, W_TYPE *weight, DistT *dist, Worklist2 &inwl, Worklist2 &outwl) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(inwl.pop_id(id, vertex)) {
		size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(true) {
		if(size > BLKSIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1)
			break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = vertex;
			inwl.dwl[id] = -1;
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
				process_edge(sh_vertex, edge, column_indices, weight, dist, outwl);
			}
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define NUM_WARPS (BLKSIZE / WARP_SIZE)
__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, W_TYPE *weight, DistT *dist, Worklist2 &inwl, Worklist2 &outwl) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int vertex;
	if(inwl.pop_id(id, vertex)) {
		if (vertex != -1)
			size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = vertex;
			inwl.dwl[id] = -1;
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
				process_edge(winner, edge, column_indices, weight, dist, outwl);
			}
		}
	}
}

__global__ void sssp_kernel(int m, int *row_offsets, int *column_indices, W_TYPE *weight, DistT *dist, Worklist2 inwl, Worklist2 outwl) {
	//expandByCta(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	//expandByWarp(m, row_offsets, column_indices, weight, dist, inwl, outwl);
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLKSIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ int src[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighborsize = 0;
	int neighboroffset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(inwl.pop_id(id, vertex)) {	  
		if(vertex != -1) {
			neighboroffset = row_offsets[vertex];
			neighborsize = row_offsets[vertex + 1] - neighboroffset;
		}
	}
	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	int done = 0;
	int neighborsdone = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			src[scratch_offset + i - done] = vertex;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			process_edge(src[threadIdx.x], edge, column_indices, weight, dist, outwl);
		}
		total_edges -= BLKSIZE;
		done += BLKSIZE;
	}
}

__global__ void insert(Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		int item = 0;
		inwl.push(item);
	}
	return;
}

void SSSPSolver(int m, int nnz, int *d_row_offsets, int *d_column_indices, W_TYPE *d_weight, DistT *d_dist) {
	DistT zero = 0;
	int iter = 0;
	double starttime, endtime, runtime;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	//initialize <<<nblocks, nthreads>>> (d_dist, m);
	//CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice));

	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	const size_t max_blocks = maximum_residency(sssp_kernel, BLKSIZE, 0);
	printf("Solving, max_blocks=%d, nthreads=%d\n", max_blocks, nthreads);
	starttime = rtclock();
	insert<<<1, BLKSIZE>>>(*inwl);
	nitems = inwl->nitems();
	while(nitems > 0) {
		++ iter;
		unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
		printf("iteration=%d, nblocks=%d, wlsz=%d\n", iter, nblocks, nitems);
		sssp_kernel<<<nblocks, BLKSIZE>>>(m, d_row_offsets, d_column_indices, d_weight, d_dist, *inwl, *outwl);
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	};
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, runtime);
	return;
}
