// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define PR_VARIANT "delta"
#include "pr.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#include <thrust/sequence.h>

typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void initialize(int m, ScoreT *sums, ScoreT *deltas, ScoreT *contrib, ScoreT init_score) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		sums[id] = 0;
		contrib[id] = 0;
		deltas[id] = init_score;
	}
}

__global__ void push_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *deltas, ScoreT *sums, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if(queue.pop_id(tid, src)) {
		IndexT row_begin = row_offsets[src];
		IndexT row_end = row_offsets[src+1];
		int degree = row_end - row_begin;
		ScoreT contribution = deltas[src] / (ScoreT)degree;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT dst = column_indices[offset];
			atomicAdd(&sums[dst], contribution);
		}
	}
}

__global__ void contrib(int m, ScoreT *deltas, int *degrees, ScoreT *outgoing_contrib) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < m) {
		outgoing_contrib[n] = deltas[n] / degrees[n];
	}
}

#if 0
__global__ void pull_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *sums, ScoreT *outgoing_contrib) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst < m) {
		IndexT row_begin = row_offsets[dst];
		IndexT row_end = row_offsets[dst+1];
		ScoreT incoming_total = 0;
		for (IndexT offset = row_begin; offset < row_end; ++ offset) {
			IndexT src = column_indices[offset];
			incoming_total += outgoing_contrib[src];
		}
		sums[dst] = incoming_total;
	}
}
#else
__global__ void pull_step(int m, IndexT *row_offsets, IndexT *column_indices, ScoreT *sums, ScoreT *outgoing_contrib) {
	__shared__ ScoreT sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int dst = warp_id; dst < m; dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[dst + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		ScoreT sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int src = column_indices[offset];
			sum += outgoing_contrib[src];
		}
		// store local sum in shared memory,
		// and reduce local sums to global sum
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(thread_lane == 0) sums[dst] += sdata[threadIdx.x];
	}
}
#endif

__global__ void update_first(int m, ScoreT *scores, ScoreT *sums, ScoreT *deltas, ScoreT base_score, ScoreT init_score, Worklist2 queue) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	//int to_add = 0; // add to the frontier or not
	if (u < m) {
		deltas[u] = base_score + kDamp * sums[u];
		deltas[u] -= init_score;
		scores[u] += deltas[u];
		sums[u] = 0;
		if (fabs(deltas[u]) > epsilon2 * scores[u])
			queue.push(u);
			//to_add = 1;
	}
	//queue.push_1item<BlockScan>(to_add, u, BLOCK_SIZE);
}

__global__ void update(int m, ScoreT *scores, ScoreT *sums, ScoreT *deltas, ScoreT base_score, ScoreT init_score, Worklist2 queue) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	//int to_add = 0; // add to the frontier or not
	if (u < m) {
		deltas[u] = kDamp * sums[u];
		scores[u] += deltas[u];
		sums[u] = 0;
		if (fabs(deltas[u]) > epsilon2 * scores[u])
			queue.push(u);
			//to_add = 1;
	}
	//queue.push_1item<BlockScan>(to_add, u, BLOCK_SIZE);
}

__global__ void l1norm(int m, ScoreT *deltas, float *diff) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0;
	if(u < m) local_diff += fabs(deltas[u]);
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

__global__ void init_queue(int m, Worklist2 &queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m) queue.d_queue[id] = id;
	return;
}

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	IndexT *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	int *d_degrees;
	ScoreT *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(IndexT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_degrees, degrees, m * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
	ScoreT *d_sums, *d_deltas, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, m * sizeof(ScoreT)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	Worklist2 frontier(m);
	int nitems = m;
	thrust::sequence(thrust::device, frontier.d_queue, frontier.d_queue + m);
	frontier.set_index(m);
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, d_sums, d_deltas, d_contrib, init_score);
	CudaTest("initializing failed");
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		if (nitems < m/8) {
			printf("push:");
			push_step <<<nblocks, nthreads>>> (m, d_out_row_offsets, d_out_column_indices, d_deltas, d_sums, frontier);
		} else {
			printf("pull:");
			contrib <<<nblocks, nthreads>>>(m, d_deltas, d_degrees, d_contrib);
			pull_step <<<nblocks, nthreads>>> (m, d_in_row_offsets, d_in_column_indices, d_sums, d_contrib);
		}
		CudaTest("solving kernel push/pull failed");
		frontier.reset();
		if(iter == 1) update_first <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_deltas, base_score, init_score, frontier);
		else update <<<nblocks, nthreads>>> (m, d_scores, d_sums, d_deltas, base_score, init_score, frontier);
		CudaTest("solving kernel update failed");
		nitems = frontier.nitems();
		//printf(" queue_size=%d", nitems);
		l1norm <<<nblocks, nthreads>>> (m, d_deltas, d_diff);
		CudaTest("solving kernel l1norm failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %lf\n", iter, h_diff);
		if(h_diff < EPSILON) break;
	} while (nitems > 0 && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
