#define PR_VARIANT "fusion"
#include <cub/cub.cuh>
#include "pr.h"
#include "timer.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;

__global__ void initialize(int m, ScoreT *scores, ScoreT init_score) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		scores[id] = init_score;
		//active[id] = true;
	}
}

__global__ void pr_kernel(int m, int *row_offsets, int *column_indices, ScoreT *scores, int *degree, ScoreT *outgoing_contrib, float *diff, bool *active, ScoreT base_score, GlobalBarrier gb) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_vertices_per_thread = (m - 1) / (gridDim.x * blockDim.x) + 1;
	int total_inputs = num_vertices_per_thread;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if (src < m) {
			outgoing_contrib[src] = scores[src] / degree[src];
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
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			local_diff += abs(scores[src] - old_score);
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) {
		atomicAdd(diff, block_sum);
	}
}

void PRSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *out_row_offsets, int *out_column_indices, int *h_degree, ScoreT *h_scores) {
	float *d_diff, h_diff;
	ScoreT *d_contrib;
	bool *d_active;
	Timer t;
	int iter = 0;
	int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices, *d_degree;
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_active, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));

	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	initialize <<<nblocks, nthreads>>> (m, d_scores, init_score);
	CudaTest("initializing failed");

	int max_blocks = 5;
	max_blocks = maximum_residency(pr_kernel, nthreads, 0);
    cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	int nSM = deviceProp.multiProcessorCount;
	nblocks = nSM * max_blocks;
	GlobalBarrierLifetime gb;
	gb.Setup(nblocks);
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
	do {
		++iter;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		pr_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_degree, d_contrib, d_diff, d_active, base_score, gb);
		CudaTest("solving kernel failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degree));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	//CUDA_SAFE_CALL(cudaFree(d_active));
	return;
}
