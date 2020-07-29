// Copyright (c) 2020 MIT
#include "tc.h"
#include "timer.h"
#include "gpu_graph.h"
#include <cub/cub.cuh>

__global__ void triangle_count(int m, CSRGraph g, int *total) {
	typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	int local_total = 0;
	for(int u = warp_id; u < m; u += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = g.edge_begin(u + thread_lane);
		auto u_begin = ptrs[warp_lane][0];
		auto u_end   = ptrs[warp_lane][1];
		for (auto e = u_begin + thread_lane; e < u_end; e += WARP_SIZE) {
			auto v = g.getEdgeDst(e);
			auto v_begin = g.edge_begin(v);
			auto v_end = g.edge_end(v);
			auto it = u_begin;
			for (auto e_dst = v_begin; e_dst < v_end; ++ e_dst) {
				auto w = g.getEdgeDst(e_dst);
				while (g.getEdgeDst(it) < w && it != u_end) it ++;
				if (it != u_end && g.getEdgeDst(it) == w) local_total += 1;
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

  const int nthreads = BLOCK_SIZE;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  const int nSM = deviceProp.multiProcessorCount;
  const int max_blocks_per_SM = maximum_residency(triangle_count, nthreads, 0);
  const int max_blocks = max_blocks_per_SM * nSM;
  int nblocks = DIVIDE_INTO(m, WARPS_PER_BLOCK);
  if(nblocks > max_blocks) nblocks = max_blocks;
  printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

  Timer t;
  t.Start();
  triangle_count<<<nblocks, nthreads>>>(m, gpu_graph, d_total);
  CudaTest("solving triangle_count kernel failed");
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\truntime [cuda_topo_warp] = %f sec \n", t.Seconds());
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
  total = (uint64_t)h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

