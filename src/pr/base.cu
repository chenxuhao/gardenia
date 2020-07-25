// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>

#define FUSED 0
#define PR_VARIANT "pull"

typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
}

__global__ void pull_step(int m, const uint64_t *row_offsets, 
                          const VertexId *column_indices, 
                          ScoreT *sums, ScoreT *outgoing_contrib) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst < m) {
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			//IndexT src = column_indices[offset];
			IndexT src = __ldg(column_indices+offset);
			//incoming_total += outgoing_contrib[src];
			incoming_total += __ldg(outgoing_contrib+src);
		}
		sums[dst] = incoming_total;
	}
}

__global__ void l1norm(int m, ScoreT *scores, ScoreT *sums, 
                       float *diff, ScoreT base_score) {
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

// pull operation needs incoming neighbor list
__global__ void pull_fused(int m, const uint64_t *row_offsets, 
                           const VertexId *column_indices,
                           ScoreT *scores, ScoreT *outgoing_contrib, 
                           float *diff, ScoreT base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	if (src < m) {
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src + 1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT dst = column_indices[offset];
			incoming_total += outgoing_contrib[dst];
		}
		ScoreT old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(Graph &g, ScoreT *scores) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.in_rowptr();
  auto h_column_indices = g.in_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

  std::vector<VertexId> degrees(m);
  for (VertexId i = 0; i < m; i ++) degrees[i] = g.get_degree(i);
	int *d_degrees;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, &degrees[0], m * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores, *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	const ScoreT base_score = (1.0f - kDamp) / m;
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
		CudaTest("solving kernel contrib failed");
#if FUSED
		pull_fused<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, base_score);
#else
		pull_step<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
		l1norm<<<nblocks, nthreads>>>(m, d_scores, d_sums, d_diff, base_score);
#endif
		CudaTest("solving kernel pull failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [cuda_pull] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_degrees));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
