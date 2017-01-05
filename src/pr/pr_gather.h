#define BFS_VARIANT "gather"
#include <cub/cub.cuh>
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define EPSILON 0.001
#define MAX_ITER 19
const float kDamp = 0.85;
#define BLKSIZE 256
typedef float ScoreT;
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *score, ScoreT init_score) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		score[id] = init_score;
		//active[id] = true;
	}
}

__global__ void process(int m, ScoreT *score, int *degree, ScoreT *outgoing_contrib, bool *active) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
		//if (src < m && active[src]) {
			//outgoing_contrib[src] = degree[src]>0? score[src] / degree[src] : 0;
			outgoing_contrib[src] = score[src] / degree[src];
			//if(src<3) printf("score[%d]=%.8f, degree[%d]=%d\n", src, score[src], src, degree[src]);
		}
	}
}

// gather operation needs incoming neighbor list
__global__ void gather(int m, int *row_offsets, int *column_indices, ScoreT *score, ScoreT *contrib, float *diff, bool *active, const ScoreT base_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float error = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
		//if (src < m && active[src]) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			ScoreT incoming_total = 0;
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				incoming_total += contrib[dst];
			}
			ScoreT old_score = score[src];
			score[src] = base_score + kDamp * incoming_total;
			error += fabs(score[src] - old_score);
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree, ScoreT *d_score) {
	print_device_info(0);
	double starttime, endtime, runtime;
	float *d_diff, h_diff;
	ScoreT *d_contrib;
	bool *d_active;
	int iter = 0;
	const int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_active, m * sizeof(bool)));
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	initialize <<<nblocks, nthreads>>> (m, d_score, init_score);
	CudaTest("initializing failed");

	size_t max_blocks = 5;
	max_blocks = maximum_residency(gather, nthreads, 0);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		process<<<nblocks, nthreads>>>(m, d_score, d_degree, d_contrib, d_active);
		CudaTest("solving kernel1 failed");
		gather<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_score, d_contrib, d_diff, d_active, base_score);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iter, h_diff);
		//ScoreT *h_score = (ScoreT *) malloc(m * sizeof(ScoreT));
		//CUDA_SAFE_CALL(cudaMemcpy(h_score, d_score, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
		//for(int i = 0; i < 10; i++) printf("score[%d]=%.8f\n", i, h_score[i]);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	//CUDA_SAFE_CALL(cudaFree(d_active));
	return;
}
