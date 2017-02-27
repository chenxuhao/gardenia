// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "base"
#include "bfs.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

#ifdef TEXTURE
texture <int, 1, cudaReadModeElementType> row_offsets;
texture <int, 1, cudaReadModeElementType> column_indices;
#endif
__global__ void initialize(unsigned *dist, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

#ifdef TEXTURE
__global__ void bfs_kernel(int m, DistT *dist, Worklist2 inwl, Worklist2 outwl) {
#else
__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, Worklist2 inwl, Worklist2 outwl) {
#endif
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(inwl.pop_id(tid, src)) {
#ifdef TEXTURE
		unsigned row_begin = tex1Dfetch(row_offsets, src);
		unsigned row_end = tex1Dfetch(row_offsets, src + 1);
#else
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1];
#endif
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
#ifdef TEXTURE
			int dst = tex1Dfetch(column_indices, offset);
#else
			int dst = column_indices[offset];
#endif
			//DistT altdist = dist[src] + 1;
			if ((dist[dst] == MYINFINITY) && (atomicCAS(&dist[dst], MYINFINITY, dist[src]+1)==MYINFINITY)) {
			//if (dist[dst] == MYINFINITY) {//Not visited
			//	dist[dst] = altdist;
				assert(outwl.push(dst));
			}
		}
	}
}

__global__ void insert(int source, Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(source);
	}
	return;
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	DistT zero = 0;
	int iter = 0;
	Timer t;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;

	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));

	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));

#ifdef TEXTURE
	CUDA_SAFE_CALL(cudaBindTexture(0, row_offsets, d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, column_indices, d_column_indices, (nnz + 1) * sizeof(int)));
#endif
	//initialize <<<nblocks, nthreads>>> (d_dist, m);
	//CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	unsigned nitems = 1;
	t.Start();
	insert<<<1, nthreads>>>(source, *inwl);
	nitems = inwl->nitems();
	do {
		++ iter;
		nblocks = (nitems - 1) / nthreads + 1;
		printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iter, nblocks, nthreads, nitems);
#ifdef TEXTURE
		bfs_kernel <<<nblocks, nthreads>>> (m, d_dist, *inwl, *outwl);
#else
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, *inwl, *outwl);
#endif
		CudaTest("solving failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
