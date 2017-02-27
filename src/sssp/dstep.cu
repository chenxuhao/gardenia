// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "delta-stepping"
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
/*
[1] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, “Work-efficient
	parallel gpu methods for single-source shortest paths,” in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349–359, May 2014
*/

__global__ void initialize(int m, DistT *dist) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__global__ void insert(int source, Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(source);
	}
	return;
}

/**
 * @brief Delta-Stepping SSSP kernel entry point.
 *
 * @param[in] m                 Number of vertices
 * @param[in] d_row_offsets     Device pointer of VertexId to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue
 * @param[in] d_weight          Device pointer of DistT to the edge weight queue
 * @param[out]d_dist            Device pointer of DistT to the distance queue
 * @param[in] d_in_queue        Device pointer of VertexId to the incoming frontier queue
 * @param[out]d_out_queue       Device pointer of VertexId to the outgoing frontier queue
 * @param[out]d_far_queue       Device pointer of VertexId to the far frontier queue
 */
 #define NUM_LOCAL_BINS 1
 #define LOCAL_BINS_SIZE BLKSIZE
 typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__global__ void delta_stepping(int m, int delta, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, Worklist2 in_queue, Worklist2 near_queue, Worklist2 far_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ int local_bins[NUM_LOCAL_BINS][LOCAL_BINS_SIZE];
	//__shared__ int bin_tails[NUM_LOCAL_BINS];
	//bin_tails[tid] = 0;
	//__syncthreads();
	int src;
	if(in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			DistT old_dist = dist[dst];
			DistT new_dist = dist[src] + weight[offset];
			if (new_dist < old_dist) {
				bool changed_dist = true;
				while (atomicCAS(&dist[dst], old_dist, new_dist) != old_dist) {
					old_dist = dist[dst];
					if (old_dist <= new_dist) {
						changed_dist = false;
						break;
					}
				}
				if (changed_dist) {
					size_t dest_bin = new_dist/delta; // calculate which bin to push
					if (dest_bin >= NUM_LOCAL_BINS) { // too far away
						far_queue.push(dst); // push into the far queue
					} else {
						//local_bins[dest_bin][bin_tails[dest_bin]++] = dst; // push into the specific local bin
						near_queue.push(dst); // push into the near queue
					}
				}
			}
		}
	}
	/*
	// find the smallest non-empty bin
	__shared__ int min_bin_index;
	if (tid == 0) min_bin_index = 0;
	if (bin_tails[tid] != 0) {
		min_bin_index = atomicMin(&min_bin_index, tid);
	}
	// push the vertices in the smallest non-empty local bin into the global shared frontier
	if(tid<bin_tails[min_bin_index])
		assert(out_queue.push(local_bins[min_bin_index][tid]));
	*/
}

/**
 * @brief delta-stepping GPU SSSP entry point.
 *
 * @param[in] m                 Number of vertices
 * @param[in] h_row_offsets     Host pointer of VertexId to the row offsets queue
 * @param[in] h_column_indices  Host pointer of VertexId to the column indices queue
 * @param[in] h_weight          Host pointer of DistT to the edge weight queue
 * @param[out]h_dist            Host pointer of DistT to the distance queue
 */
void SSSPSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist) {
	DistT zero = 0;
	int step = 1, iter = 0;
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
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 wl1(nnz), wl2(nnz), wl3(nnz);
	Worklist2 *in_frontier = &wl1, *near_frontier = &wl2, *far_frontier = &wl3;
	int in_size = 1, near_size = 0, far_size = 0;
	int delta = 1;
	int max_blocks = maximum_residency(delta_stepping, nthreads, 0);
	printf("Launching CUDA SSSP solver (%d CTAs/SM, %d threads/CTA) ...\n", max_blocks, nthreads);

	Timer t;
	t.Start();
	insert<<<1, nthreads>>>(source, *in_frontier);
	in_size = in_frontier->nitems();
	do {
		iter ++;
		nblocks = (in_size - 1) / nthreads + 1;
		//printf("iteration=%d, nblocks=%d, nthreads=%d, frontier_size=%d, near_size=%d, far_size=%d, step=%d\n", iter, nblocks, nthreads, in_size, near_size, far_size, step);
		delta_stepping<<<nblocks, nthreads>>>(m, delta*step, d_row_offsets, d_column_indices, d_weight, d_dist, *in_frontier, *near_frontier, *far_frontier);
		CudaTest("solving failed");
		near_size = near_frontier->nitems();
		far_size = far_frontier->nitems();
		Worklist2 *tmp = in_frontier;
		if(near_size > 0) {
			in_size = near_size;
			in_frontier = near_frontier;
			near_frontier = tmp;
			near_frontier->reset();
		} else {
			++ step;
			in_size = far_size;
			in_frontier = far_frontier;
			far_frontier = tmp;
			far_frontier->reset();
		}
	} while(in_size > 0);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
