#define PR_VARIANT "gather"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
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
				//if(src==0) printf("dst=%d\n", dst);
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

void PRSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *out_row_offsets, int *out_column_indices, int *h_degree, ScoreT *h_score) {
	print_device_info(0);
	Timer t;
	float *d_diff, h_diff;
	ScoreT *d_contrib;
	bool *d_active;
	int iter = 0;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices, *d_degree;
	ScoreT *d_score;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_score, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_active, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	printf("base_score=%.8f, init_score=%.8f\n", base_score, init_score);
	initialize <<<nblocks, nthreads>>> (m, d_score, init_score);
	CudaTest("initializing failed");
	//CUDA_SAFE_CALL(cudaMemcpy(h_score, d_score, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	//for(int i = 0; i < 5; i++) printf("score[%d]=%.8f\n", i, h_score[i]);

	size_t max_blocks = 5;
	max_blocks = maximum_residency(gather, nthreads, 0);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
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
		//CUDA_SAFE_CALL(cudaMemcpy(h_score, d_score, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
		//for(int i = 0; i < 5; i++) printf("score[%d]=%.8f\n", i, h_score[i]);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_score, d_score, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degree));
	CUDA_SAFE_CALL(cudaFree(d_score));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	//CUDA_SAFE_CALL(cudaFree(d_active));
	return;
}
