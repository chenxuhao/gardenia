#define BFS_VARIANT "scatter"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define EPSILON 0.001
#define MAX_ITER 19
#define BLKSIZE 256
const float kDamp = 0.85;
typedef float ScoreT;
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *cur_score, ScoreT *next_score, ScoreT init_score, ScoreT base_score) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_score[id] = init_score;
		next_score[id] = base_score;
	}
}

__global__ void scatter(int m, int *row_offsets, int *column_indices, ScoreT *cur_score, ScoreT *next_score) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			int degree = row_end - row_begin;
			ScoreT value = kDamp * cur_score[src] / (ScoreT)degree;
			//if(src<3) printf("score[%d]=%.8f, degree[%d]=%d\n", src, cur_score[src], src, degree);
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				atomicAdd(&next_score[dst], value);
			}
		}
	}
}
/*
__global__ void reduce(int m, ScoreT *cur_score, ScoreT *next_score, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			float local_diff = fabs(next_score[src] - cur_score[src]);
			atomicAdd(diff, local_diff);
			cur_score[src] = next_score[src];
			next_score[src] = (1.0f - kDamp) / m;
		}
	}
}
//*/
///*
__global__ void reduce(int m, ScoreT *cur_score, ScoreT *next_score, float *diff) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			local_diff += fabs(next_score[src] - cur_score[src]);
			cur_score[src] = next_score[src];
			next_score[src] = (1.0f - kDamp) / m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}
//*/
void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree, ScoreT *d_cur_score) {
	double starttime, endtime, runtime;
	float *d_diff, h_diff;
	ScoreT *d_next_score;
	int iter = 0;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_score, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	const ScoreT init_score = 1.0f / m;
	const ScoreT base_score = (1.0f - kDamp) / m;
	initialize <<<nblocks, nthreads>>> (m, d_cur_score, d_next_score, init_score, base_score);
	CudaTest("initializing failed");

	const size_t max_blocks = maximum_residency(scatter, nthreads, 0);
	//const size_t max_blocks = 5;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		scatter <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_score, d_next_score);
		CudaTest("solving kernel1 failed");
		reduce <<<nblocks, nthreads>>> (m, d_cur_score, d_next_score, d_diff);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_next_score));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
