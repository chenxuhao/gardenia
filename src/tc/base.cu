// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cub/cub.cuh>
#include "tc.h"
#include "timer.h"
#include "gpu_graph.h"

__global__ void ordered_count(int m, CSRGraph g, int *total) {
	typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int local_total = 0;
	if (u < m) {
		auto begin_u = g.edge_begin(u);
		auto end_u = g.edge_end(u); 
		for (auto off_u = begin_u; off_u < end_u; ++ off_u) {
			auto v = g.getEdgeDst(off_u);
			auto begin_v = g.edge_begin(v);
			auto end_v = g.edge_end(v);
      auto it = begin_u;
			for (auto off_v = begin_v; off_v < end_v; ++ off_v) {
				auto w = g.getEdgeDst(off_v);
				while (g.getEdgeDst(it) < w && it < end_u) it ++;
				if (it != end_u && g.getEdgeDst(it) == w) local_total += 1;
			}
		}
	}
	int block_total = BlockReduce(temp_storage).Sum(local_total);
	if(threadIdx.x == 0) atomicAdd(total, block_total);
}

void TCSolver(Graph &g, uint64_t &total) {
  //print_device_info(0);
  CSRGraph gpu_graph(g);
  auto m = g.V();
  int zero = 0;
  int h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(int), cudaMemcpyHostToDevice));

  int nthreads = BLOCK_SIZE;
  int nblocks = (m - 1) / nthreads + 1;
  int max_blocks = maximum_residency(ordered_count, nthreads, 0);
  printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

  Timer t;
  t.Start();
  ordered_count<<<nblocks, nthreads>>>(m, gpu_graph, d_total);
  CudaTest("solving failed");
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\truntime [cuda_base] = %f sec\n", t.Seconds());
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
  total = (uint64_t)h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

