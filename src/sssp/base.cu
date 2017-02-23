// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "base"
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void initialize(int m, DistT *dist) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__global__ void sssp_kernel(int m, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, Worklist2 inwl, Worklist2 outwl) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(inwl.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			DistT wt = weight[offset];
			DistT altdist = dist[src] + wt;
			if (altdist < dist[dst]) {
				DistT olddist = atomicMin(&dist[dst], altdist);
				if (altdist < olddist) { // update successfully
					assert(outwl.push(dst));
				}
			}
		}
	}
}

__global__ void insert(Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(0);
	}
	return;
}

void SSSPSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist) {
	DistT zero = 0;
	int iteration = 0;
	Timer t;
	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	//initialize <<<nblocks, nthreads>>> (m, d_dist);
	//CudaTest("initializing failed");

	int *d_row_offsets, *d_column_indices;
	DistT *d_weight;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(DistT), cudaMemcpyHostToDevice));
	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	//const size_t max_blocks = maximum_residency(sssp_kernel, nthreads, 0);
	t.Start();
	insert<<<1, nthreads>>>(*inwl);
	nitems = inwl->nitems();
	do {
		++iteration;
		nblocks = (nitems - 1) / nthreads + 1;
		//printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iteration, nblocks, nthreads, nitems);
		sssp_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, d_dist, *inwl, *outwl);
		CudaTest("solving failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iteration);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
