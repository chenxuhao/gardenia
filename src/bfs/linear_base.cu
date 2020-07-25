// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "bfs.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

__global__ void bfs_kernel(int m, const uint64_t *row_offsets, 
                           const IndexT *column_indices, 
                           DistT *dists, Worklist2 in_queue, 
                           Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((dists[dst] == MYINFINITY) && 
          (atomicCAS(&dists[dst], MYINFINITY, dists[src]+1) == MYINFINITY)) {
				assert(out_queue.push(dst));
			}
		}
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

void BFSSolver(Graph &g, int source, DistT *h_dists) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

  DistT zero = 0;
  DistT * d_dists;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dists, m * sizeof(DistT)));
  CUDA_SAFE_CALL(cudaMemcpy(d_dists, h_dists, m * sizeof(DistT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_dists[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Worklist2 queue1(m), queue2(m);
  Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
  int iter = 0;
  int nitems = 1;
  int nthreads = BLOCK_SIZE;
  int nblocks = (m - 1) / nthreads + 1;
  printf("Launching CUDA BFS solver (%d threads/CTA) ...\n", nthreads);

  Timer t;
  t.Start();
  insert<<<1, nthreads>>>(source, *in_frontier);
  nitems = in_frontier->nitems();
  do {
    ++ iter;
    nblocks = (nitems - 1) / nthreads + 1;
    //printf("iteration %d: frontier_size = %d\n", iter, nitems);
    bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, 
        d_dists, *in_frontier, *out_frontier);
    CudaTest("solving bfs_kernel failed");
    nitems = out_frontier->nitems();
    Worklist2 *tmp = in_frontier;
    in_frontier = out_frontier;
    out_frontier = tmp;
    out_frontier->reset();
  } while (nitems > 0);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\titerations = %d.\n", iter);
  printf("\truntime [cuda_linear_base] = %f ms.\n", t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_dists, m * sizeof(DistT), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_row_offsets));
  CUDA_SAFE_CALL(cudaFree(d_column_indices));
  CUDA_SAFE_CALL(cudaFree(d_dists));
  return;
}

