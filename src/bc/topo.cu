// Copyright (c) 2016, Xuhao Chen
#define BC_VARIANT "topo"
#include "bc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"
#include "timer.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

__global__ void initialize(int m, int source, ScoreT *scores, int *path_counts, int *depths, ScoreT *deltas, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		scores[id] = 0;
		deltas[id] = 0;
		expanded[id] = false;
		if(id == source) {
			visited[id] = true;
			path_counts[id] = 1;
			depths[id] = 0;
		} else {
			visited[id] = false;
			path_counts[id] = 0;
			depths[id] = -1;
		}
	}
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(int m, int *row_offsets, int *column_indices, int *path_counts, int *depths, int depth, bool *changed, bool *visited, bool *expanded, int *nitems, int *queue, int queue_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	//for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
	int src = tid;
	if(src < m && visited[src] && !expanded[src]) {
		expanded[src] = true;
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			//int depth = depths[src] + 1;
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
				int pos = atomicAdd(nitems, 1);
				queue[queue_len + pos] = dst;
				*changed = true;
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num) {
		int src = frontiers[start + id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			//if(depths[dst] == depths[src] + 1) {
			if(depths[dst] == depth + 1) {
				delta_src += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		deltas[src] = delta_src;
		scores[src] += deltas[src];
	}
}

__global__ void bc_update(int m, int *depths, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(depths[id] != -1 && !visited[id])
			visited[id] = true;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) {
		scores[tid] = scores[tid] / (max_score);
	}
}

void BCSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores) {
	print_device_info(0);
	Timer t;
	int zero = 0;
	int depth = 0;
	vector<int> depth_index;
	int *d_row_offsets, *d_column_indices;
	ScoreT *d_scores, *d_deltas;
	int *d_path_counts, *d_depths, *d_frontiers;
	bool *d_changed, h_changed, *d_visited, *d_expanded;
	int *d_nitems, h_nitems = 1;

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (m+1)));

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nitems, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	//printf("Copy data to device...\n");
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));

	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	//printf("Initializing data on device...\n");
	initialize <<<nblocks, nthreads>>> (m, source, d_scores, d_path_counts, d_depths, d_deltas, d_visited, d_expanded);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice));

	int frontiers_len = 0;
	int max_blocks = maximum_residency(bc_forward, nthreads, 0);
	depth_index.push_back(0);
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", max_blocks, nthreads);
	t.Start();

	do {
		depth++;
		h_changed = false;
		printf("iteration=%d, frontire_size=%d\n", depth, h_nitems);
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_nitems, &zero, sizeof(int), cudaMemcpyHostToDevice));
		frontiers_len += h_nitems;
		depth_index.push_back(frontiers_len);
		bc_forward<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_path_counts, d_depths, depth, d_changed, d_visited, d_expanded, d_nitems, d_frontiers, frontiers_len);
		CudaTest("solving bc_forward failed");
		bc_update <<<nblocks, nthreads>>> (m, d_depths, d_visited);
		CudaTest("solving bc_update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&h_nitems, d_nitems, sizeof(int), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		h_nitems = depth_index[d+1] - depth_index[d];
		thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		nblocks = (h_nitems - 1) / nthreads + 1;
		printf("Reverse: depth=%d, frontier_size=%d\n", d, h_nitems);
		bc_reverse<<<nblocks, nthreads>>>(h_nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving kernel2 failed");
	}
	
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	printf("\nStart calculating the maximum score\n");
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	//h_max_score = *max_element(h_scores, h_scores+m);
	//for (int n = 0; n < m; n ++) h_scores[n] = h_scores[n] / h_max_score;
	//std::cout << "max_score = " << h_max_score << "\n";
	printf("\nStart normalizing scores\n");
	nthreads = 512;
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

