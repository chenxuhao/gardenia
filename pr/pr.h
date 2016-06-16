#define BFS_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void initialize(float *cur_pagerank, float *next_pagerank, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_pagerank[id] = 1.0f / (float)m;
		next_pagerank[id] = 1.0f / (float)m;
	}
}

__global__ void update_neighbors(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			unsigned degree = row_end - row_begin;
			float value = 0.85 * cur_pagerank[src] / (float)degree;
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				next_pagerank[dst] += value;
			}
		}
	}
}
/*
__global__ void self_update(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			float local_diff = abs(next_pagerank[src] - cur_pagerank[src]);
			atomicAdd(diff, local_diff);
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = 0.15 / (float)m;
		}
	}
}
*/
#include <cub/cub.cuh>
__global__ void self_update(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockReduce<float, 256> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			local_diff += abs(next_pagerank[src] - cur_pagerank[src]);
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = 0.15 / (float)m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, foru* d_weight, int nSM) {
	unsigned zero = 0;
	float *d_diff, h_diff, e = 0.1;
	float *d_cur_pagerank, *d_next_pagerank;
	double starttime, endtime, runtime;
	int iteration = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_cur_pagerank, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_pagerank, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	initialize <<<nblocks, nthreads>>> (d_cur_pagerank, d_next_pagerank, m);
	CudaTest("initializing failed");

	//const size_t max_blocks_1 = maximum_residency(update_neighbors, nthreads, 0);
	const size_t max_blocks = maximum_residency(self_update, nthreads, 0);
	//printf("max_blocks_1=%d, max_blocks=%d\n", max_blocks_1, max_blocks);
	//const size_t max_blocks = 5;
	if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
	printf("Solving, nSM=%d, max_blocks=%d, nblocks=%d, nthreads=%d\n", nSM, max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iteration;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		update_neighbors <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank);
		self_update <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank, d_diff);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iteration, h_diff);
	} while (h_diff > e && iteration < 20);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
