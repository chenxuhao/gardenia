// Copyright (c) 2016, Xuhao Chen
#include "bc.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <vector>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#define BC_VARIANT "topo_base"

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
__global__ void bc_forward(int m, const uint64_t *row_offsets, 
                           const IndexT *column_indices, 
                           int *path_counts, int *depths, int depth, 
                           bool *changed, bool *visited, bool *expanded, 
                           int *nitems, int *queue, int queue_len) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && visited[src] && !expanded[src]) {
		expanded[src] = true;
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
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
__global__ void bc_reverse(int num, const uint64_t *row_offsets, 
                           const IndexT *column_indices, 
                           int start, int *frontiers, 
                           ScoreT *scores, int *path_counts, 
                           int *depths, int depth, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num) {
		int src = frontiers[start + id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
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
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

void BCSolver(Graph &g, int source, ScoreT *h_scores) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
	//print_device_info(0);
	int zero = 0;
	uint64_t *d_row_offsets;
  VertexId *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
	
	ScoreT *d_scores, *d_deltas;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	int *d_path_counts, *d_depths, *d_frontiers;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (m+1)));
	bool *d_changed, h_changed, *d_visited, *d_expanded;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	int *d_nitems, h_nitems = 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nitems, sizeof(int)));

	int depth = 0;
	int frontiers_len = 0;
	vector<int> depth_index;
	depth_index.push_back(0);
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, source, d_scores, d_path_counts, d_depths, d_deltas, d_visited, d_expanded);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		depth++;
		h_changed = false;
		//printf("iteration=%d, frontire_size=%d\n", depth, h_nitems);
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
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		h_nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		nblocks = (h_nitems - 1) / nthreads + 1;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, h_nitems);
		bc_reverse<<<nblocks, nthreads>>>(h_nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving bc_reverse failed");
	}
	
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	//printf("\nStart calculating the maximum score\n");
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	//h_max_score = *max_element(h_scores, h_scores+m);
	//for (int n = 0; n < m; n ++) h_scores[n] = h_scores[n] / h_max_score;
	//std::cout << "max_score = " << h_max_score << "\n";
	//printf("\nStart normalizing scores\n");
	nthreads = 512;
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

