#define PR_VARIANT "fusion"
#include <cub/cub.cuh>
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define EPSILON 0.001
#define MAX_ITER 19
#define BLKSIZE 256
const float kDamp = 0.85;
typedef float ScoreT;
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *score, ScoreT init_score) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		score[id] = init_score;
		//active[id] = true;
	}
}

__global__ void pr_kernel(int m, int *row_offsets, int *column_indices, ScoreT *score, int *degree, ScoreT *outgoing_contrib, float *diff, bool *active, ScoreT base_score, GlobalBarrier gb) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_vertices_per_thread = (m - 1) / (gridDim.x * blockDim.x) + 1;
	int total_inputs = num_vertices_per_thread;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
			outgoing_contrib[src] = score[src] / degree[src];
		}
	}
	gb.Sync();
	float local_diff = 0.0f;
	total_inputs = num_vertices_per_thread;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			ScoreT incoming_total = 0;
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = score[src];
			score[src] = base_score + kDamp * incoming_total;
			local_diff += abs(score[src] - old_score);
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) {
		atomicAdd(diff, block_sum);
	}
}

void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree, ScoreT *d_score) {
	float *d_diff, h_diff;
	ScoreT *d_contrib;
	bool *d_active;
	double starttime, endtime, runtime;
	int iter = 0;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_active, m * sizeof(bool)));
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	initialize <<<nblocks, nthreads>>> (m, d_score, init_score);
	CudaTest("initializing failed");

	size_t max_blocks = 5;
	max_blocks = maximum_residency(pr_kernel, nthreads, 0);
    cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	nblocks = nSM * max_blocks;
	GlobalBarrierLifetime gb;
	gb.Setup(nblocks);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iter;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		pr_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_score, d_degree, d_contrib, d_diff, d_active, base_score, gb);
		CudaTest("solving kernel failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	//CUDA_SAFE_CALL(cudaFree(d_active));
	return;
}
