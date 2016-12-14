// Copyright (c) 2016, Xuhao Chen

#define BC_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
//#include "bitmap.h"
#include "worklistc.h"
/*
Gardenia Benchmark Suite
Kernel: Will return array of approx betweenness centrality scores for each vertex (BC)
Author: Xuhao Chen

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

using namespace std;
typedef float ScoreT;

__global__ void initialize(int m, ScoreT *scores, int *path_counts, int *depths) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		scores[id] = 0;
		path_counts[id] = 0;
		depths[id] = -1;
	}
}

__global__ void bc_forward(int *row_offsets, int *column_indices, ScoreT *scores, int *path_counts, int *depths, int depth, Worklist2 inwl, Worklist2 outwl) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(inwl.pop_id(tid, src)) {
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1]; 
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth))) {
				assert(outwl.push(dst));
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

__global__ void bc_reverse(int *row_offsets, int *column_indices, ScoreT *scores, int *path_counts, Worklist2 frontier, int *depths, int depth, ScoreT *deltas) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(frontier.pop_id(tid, src)) {
		ScoreT delta_src = 0;
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1];
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(depths[src] == depth - 1)
				delta_src += path_counts[src] / path_counts[dst] * (1 + deltas[dst]);
		}
		deltas[src] = delta_src;
		scores[src] += delta_src;
	}
}

__global__ void insert(Worklist2 inwl, int src) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(src);
	}   
	return;
}

void Brandes(int m, int nnz, int *row_offsets, int *column_indices, int *degree) {
	ScoreT *h_scores, *d_scores, *d_deltas;
	int *d_path_counts, *d_depths;
	double starttime, endtime, runtime;
	int depth = 0;
	h_scores = (ScoreT *)malloc(m * sizeof(ScoreT));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	const size_t max_blocks = maximum_residency(bc_forward, nthreads, 0);
	initialize <<<nblocks, nthreads>>> (m, d_scores, d_path_counts, d_depths);
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int source = 0;
	int nitems = 1;
	vector<Worklist2 > depth_index;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();

	insert<<<1, nthreads>>>(*inwl, source);
	do {
		nblocks = (nitems - 1) / nthreads + 1;
		depth++;
		depth_index.push_back(*inwl);
		bc_forward<<<nblocks, nthreads>>>(row_offsets, column_indices, d_scores, d_path_counts, d_depths, depth, *inwl, *outwl);
		CudaTest("solving kernel1 failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		bc_reverse<<<nblocks, nthreads>>>(row_offsets, column_indices, d_scores, d_path_counts, depth_index[d], d_depths, d, d_deltas);
		CudaTest("solving kernel2 failed");
	}

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	endtime = rtclock();
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, runtime);

	ScoreT biggest_score = 0;
#pragma omp parallel for reduction(max : biggest_score)
	for (int i = 0; i < m; i++)
		biggest_score = max(biggest_score, h_scores[i]);
#pragma omp parallel for
	for (int i = 0; i < m; i++)
		h_scores[i] = h_scores[i] / biggest_score;
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
}

bool BCVerifier(int m, int *row_offsets, int *column_indices, size_t test_total) {
	size_t total = 0;
	total = total / 6;
	if (total != test_total)
		cout << total << " != " << test_total << endl;
	return total == test_total;
}

