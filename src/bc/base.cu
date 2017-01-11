// Copyright (c) 2016, Xuhao Chen
#define BC_VARIANT "linear"
#include "bc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"
#include "timer.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
/*
Gardenia Benchmark Suite
Kernel: Betweenness Centrality (BC)
Author: Xuhao Chen

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only an approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
	Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
	implementations for evaluating betweenness centrality on massive datasets."
	International Symposium on Parallel & Distributed Processing (IPDPS), 2009.
*/

__global__ void initialize(int m, ScoreT *scores, int *path_counts, int *depths, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		scores[id] = 0;
		path_counts[id] = 0;
		depths[id] = -1;
		deltas[id] = 0;
	}
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(int *row_offsets, int *column_indices, ScoreT *scores, int *path_counts, int *depths, int depth, Worklist2 inwl, Worklist2 outwl) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(inwl.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			//if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depths[src]+1)==-1)) {
				assert(outwl.push(dst));
			}
			//if (depths[dst] == depth) {
			if (depths[dst] == depths[src]+1) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < num) {
		int src = frontiers[start + tid];
		//if(num<13) printf("depth=%d: %d\n", depth, src);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		//ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			//if(src==237) printf("Before: src %d dst %d depth_src=%d, depth_dst=%d, delta_src=%.8f, delta_dst=%.8f, accu=%.8f\n", src, dst, depths[src], depths[dst], deltas[src], deltas[dst], static_cast<ScoreT>(path_counts[src]) / static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]));
			if(depths[dst] == depths[src] + 1) {
				//deltas[src] += static_cast<ScoreT>(path_counts[src]) / static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
				deltas[src] += (ScoreT)path_counts[src] / (ScoreT)path_counts[dst] * (1 + deltas[dst]);
			}
			//if(src==237) printf("After: src %d dst %d depth_src=%d, depth_dst=%d, delta_src=%.8f, delta_dst=%.8f\n", src, dst, depths[src], depths[dst], deltas[src], deltas[dst]);
		}
		//deltas[src] = delta_src;
		scores[src] += deltas[src];
		//if(src==237) printf("Vertex %d: depth=%d, out_degree=%d, path_count=%d, delta=%.8f, score=%.8f\n", src, depths[src], row_end-row_begin, path_counts[src], deltas[src], scores[src]);
	}
}

__global__ void insert(Worklist2 inwl, int src, int *path_counts, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(src);
		path_counts[src] = 1;
		depths[src] = 0;
	}
	return;
}

__global__ void push_frontier(Worklist2 inwl, int *queue, int queue_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(inwl.pop_id(tid, vertex)) {
		queue[queue_len+tid] = vertex;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT *max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m)
		scores[tid] = scores[tid] / *max_score;
}

void BCSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores, int device) {
	printf("Launching CUDA BC solver...\n");
	//print_device_info(device);
	Timer t;
	int depth = 0;
	vector<int> depth_index;
	int *d_row_offsets, *d_column_indices;//, *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));

	ScoreT *d_scores, *d_deltas;
	int *d_path_counts, *d_depths, *d_frontiers;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * 2 * m));
	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = maximum_residency(bc_forward, nthreads, 0);
	initialize <<<nblocks, nthreads>>> (m, d_scores, d_path_counts, d_depths, d_deltas);
	Worklist2 wl1(2*m), wl2(2*m);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int source = 0;
	int nitems = 1;
	int frontiers_len = 0;
	depth_index.push_back(0);
	printf("Solving, max_blocks_per_SM=%d, nthreads=%d\n", max_blocks, nthreads);
	t.Start();

	insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
	do {
		nblocks = (nitems - 1) / nthreads + 1;
		push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
		frontiers_len += nitems;
		depth_index.push_back(frontiers_len);
		//printf("Forward: depth=%d, frontire_size=%d\n", depth, nitems);
		//if(nitems < 30) inwl->display_items();
		depth++;
		bc_forward<<<nblocks, nthreads>>>(d_row_offsets, d_column_indices, d_scores, d_path_counts, d_depths, depth, *inwl, *outwl);
		CudaTest("solving kernel1 failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	//int *h_path_counts = (int *)malloc(m * sizeof(int));
	//CUDA_SAFE_CALL(cudaMemcpy(h_path_counts, d_path_counts, sizeof(int) * m, cudaMemcpyDeviceToHost));
	//int *h_depths = (int *)malloc(m * sizeof(int));
	//CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, sizeof(int) * m, cudaMemcpyDeviceToHost));
	//FILE *fp = fopen("depths.txt", "w");
	//for (int i = 0; i < m; i++) fprintf(fp, "depths[%d] = %d\n", i, h_depths[i]);
	//fclose(fp);
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	//printf("depth=%d, depth_index.size=%d\n", depth, depth_index.size());
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		nitems = depth_index[d+1] - depth_index[d];
		nblocks = (nitems - 1) / nthreads + 1;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
		bc_reverse<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving kernel2 failed");
	}
	//ScoreT *h_scores = (ScoreT *)malloc(m * sizeof(ScoreT));
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < 10; i++) printf("scores[%d] = %.8f\n", i, h_scores[i]);
	// Normalize scores
	ScoreT *d_max_score;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_score, sizeof(ScoreT)));
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, d_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	//printf("max_score = %f\n", h_max_score);
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < 10; i++) printf("scores[%d] = %.8f\n", i, h_scores[i]);
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

