// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "delta-stepping"
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
//#include <cub/cub.cuh>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
//#define COMPACT
/*
[1] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, “Work-efficient
	parallel gpu methods for single-source shortest paths,” in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349–359, May 2014
*/
/*
__global__ void initialize(int m, DistT *dist) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) dist[id] = kDistInf;
}
*/
// insert the source vertex into the work queue
__global__ void insert(int source, Worklist2 queue) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

__global__ void mark_valid(int m, int nitems, unsigned* valid) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nitems) {
		//int vertex;
		//queue.pop_id(tid, vertex);
		//if(vertex < m) {
			valid[tid] = 1;
	}
}

// compete the locks and find duplicates
__global__ void compete_locks(int m, int delta, bool valid_set, bool is_far, unsigned* valid, int *bin, int *locks, DistT *dist, DistT *f_dist, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(queue.pop_id(tid, vertex)) {
		bool is_valid = valid_set ? valid[tid] : 1;
		//bool is_valid = valid[tid];
		if(is_valid) {
			if(is_far && vertex < m) {
				DistT d = dist[vertex];
				is_valid = (bin[tid]*delta <= d);
				valid[tid] = is_valid;
				f_dist[tid] = d;
			} else if (is_far) valid[tid] = 0;
			if(is_valid && vertex < m) locks[vertex] = tid; // write the thread id into the lock
		}
	}
}

/*
__global__ void markDuplicates(int m, int nitems, unsigned* valid, int* vertices) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nitems) {
		int vertex = vertices[tid];
		bool setVal = true;
		setVal = (vertex < m && (tid == 0 || vertex > vertices[tid-1]));
		valid[tid] = (setVal ? 1 : 0);
	}
}
*/

//invalidate the duplicates (lost competing the locks) in the queue
__global__ void mark_visited(int m, bool valid_set, unsigned* valid, int *locks, DistT *dist, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(queue.pop_id(tid, vertex)) {
		assert(vertex < m);
		bool is_valid = valid_set ? valid[tid] : true;
		//bool is_valid = valid[tid];
		is_valid = (is_valid && (dist[vertex] < kDistInf) && (tid == locks[vertex]));
		valid[tid] = (is_valid ? 1 : 0);
	} else if(tid == *(queue.d_index)) valid[tid] = 0;
}

__global__ void mark_near_far(int m, bool is_relabel, int delta, int bin_min, int threshold, unsigned *valid_in, unsigned *valid_near, unsigned *valid_far, int *bin_id, int *locks, DistT *f_dist, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(queue.pop_id(tid, vertex)) {
		int bid;
		bool is_valid = is_relabel ? valid_in[tid] : true;
		if(is_valid) {
			//assert(vertex < m);
			//bid = f_dist[tid]/delta;
			bid = f_dist[vertex]/delta;
			//bin_id[tid] = bid;
			//if(is_relabel) 
			is_valid = (tid == locks[vertex]); // if wins competing the lock
		}
		valid_near[tid] = (is_valid && bid < threshold && bid >= bin_min); // mark to push into near_queue
		valid_far[tid] = (is_valid && bid >= threshold && bid < kDistInf/delta); // mark to push into far_queue
	} else if(tid == *(queue.d_index)) {
		valid_near[tid] = 0;
		valid_far[tid] = 0;
	}
}

// compact the queue, i.e. remove duplicates
__global__ void compact_queue(int nitems, unsigned* valid, int* in_queue, int* out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nitems) {
		unsigned pos = valid[tid];
		unsigned next_pos = valid[tid+1];
		bool save_out = (pos == (next_pos-1));
		if(save_out) {
			out_queue[pos] = in_queue[tid];
			//printf("thread %d: push vertex %d in %d\n", tid, in_queue[tid], pos);
		}
	}
}

// remove dupliates and split some work in the far_queue into the near_queue
__global__ void compact_queue_combo(unsigned* valid1, unsigned *valid2, int *bin_id, int low_offset, int* v1, int* b1, int up_offset, int* v2, int* b2, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid < nitems) {
	int vertex;
	if(queue.pop_id(tid, vertex)) {
		unsigned pos = valid1[tid];
		unsigned next_pos = valid1[tid+1];
		bool save_out;
		save_out = (pos == (next_pos-1));
		//int bid = bin_id[tid];
		if(save_out) {
			v1[pos+low_offset] = vertex;
			//b1[pos+low_offset] = bid;
		}

		pos = valid2[tid];
		next_pos = valid2[tid+1];
		save_out = (pos == (next_pos-1));
		if(save_out) {
			v2[pos+up_offset] = vertex;
			//b2[pos+up_offset] = bid;
		}
	}
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
 #define LOCAL_BINS_SIZE BLOCK_SIZE
// typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
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
				bool changed_dist = false;
				old_dist = atomicMin(&dist[dst], new_dist);
				if (new_dist < old_dist) { // update successfully
					changed_dist = true;
			/*	
			if (new_dist < old_dist) {
				bool changed_dist = true;
				while (atomicCAS(&dist[dst], old_dist, new_dist) != old_dist) {
					old_dist = dist[dst];
					if (old_dist <= new_dist) {
						changed_dist = false;
						break;
					}
			*/
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
void SSSPSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist, int delta) {
	DistT zero = 0;
	int step = 0, iter = 0;
	int nthreads = BLOCK_SIZE;
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
	int max_blocks = maximum_residency(delta_stepping, nthreads, 0);

	//data structure for delta-stepping
	unsigned *valid, *valid_near;//, *valid_far;
	int *vertex_locks; // used for removing duplicates in the frontier queue
	int *bin_near, *bin_far;
	DistT *frontier_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&valid, (nnz+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&valid_near, (nnz+1) * sizeof(unsigned)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&valid_far, (nnz+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&vertex_locks, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&bin_near, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&bin_far, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_dist, nnz * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemset(valid, 0, (nnz+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemset(valid_near, 0, (nnz+1) * sizeof(unsigned)));
	//CUDA_SAFE_CALL(cudaMemset(valid_far, 0, (nnz+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemset(vertex_locks, 0xFF, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(bin_near, 0, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(bin_far, 0, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(frontier_dist, 0, nnz * sizeof(DistT)));
	printf("Launching CUDA SSSP solver (%d CTAs/SM, %d threads/CTA) ...\n", max_blocks, nthreads);

	Timer t;
	t.Start();
	insert<<<1, nthreads>>>(source, *in_frontier);
	in_size = in_frontier->nitems();
	do {
		iter ++;
		nblocks = (in_size - 1) / nthreads + 1;
		printf("iteration=%d, nblocks=%d, in_size=%d, near_size=%d, far_size=%d, step=%d\n", iter, nblocks, in_size, near_size, far_size, step);
		//in_frontier->display_items();
		delta_stepping<<<nblocks, nthreads>>>(m, delta*step, d_row_offsets, d_column_indices, d_weight, d_dist, *in_frontier, *near_frontier, *far_frontier);
		CudaTest("solving failed");
		near_size = near_frontier->nitems();
		far_size = far_frontier->nitems();
		//printf("After expansion: near_size=%d, far_size=%d\n", near_size, far_size);
		if (near_size > 0) { // continue to work on the near queue
			nblocks = (near_size - 1) / nthreads + 1;
#ifdef COMPACT
			//remove duplicates in the near queue
			//printf("\tbefore near compaction: %d items\n", near_size);
			compete_locks<<<nblocks, nthreads>>>(m, delta, false, false, valid, NULL, vertex_locks, NULL, NULL, *near_frontier);
			mark_visited<<<nblocks, nthreads>>>(m, false, valid, vertex_locks, d_dist, *near_frontier);
			int num_valid_items = thrust::reduce(thrust::device, valid, valid + near_size, 0, thrust::plus<int>());
			//printf("\tafter near compaction: %d items\n", num_valid_items);
			thrust::exclusive_scan(thrust::device, valid, valid + near_size + 1, valid);
			//compact the near_queue and push the items into the in_queue
			in_frontier->reset();
			compact_queue<<<nblocks, nthreads>>>(near_size, valid, near_frontier->d_queue, in_frontier->d_queue);
			in_frontier->set_index(num_valid_items);
			near_frontier->reset();
#else
			//swap the queues
			Worklist2 *tmp = in_frontier;
			in_frontier = near_frontier;
			near_frontier = tmp;
			near_frontier->reset();
#endif
			in_size = in_frontier->nitems();
		} else if (far_size > 0) { // turn to work on the far queue since no work left in the near queue
			nblocks = (far_size - 1) / nthreads + 1;
			///*
			//thrust::sort(thrust::device, far_frontier->d_queue, far_frontier->d_queue+far_size);
			//remove duplicates in the far queue, and split some work into the near queue
			int num_near_items = 0, num_far_items = 0;
			do {
				++ step;
				//printf("\tbefore far compaction (step=%d): %d far items\n", step, far_size);
				compete_locks<<<nblocks, nthreads>>>(m, delta, false, false, valid, NULL, vertex_locks, NULL, NULL, *far_frontier);
				mark_near_far<<<nblocks, nthreads>>>(m, false, delta, 0, step, valid, valid_near, valid, NULL, vertex_locks, d_dist, *far_frontier);
				num_near_items = thrust::reduce(thrust::device, valid_near, valid_near + far_size, 0, thrust::plus<int>());
				num_far_items = thrust::reduce(thrust::device, valid, valid + far_size, 0, thrust::plus<int>());
				//printf("\tafter far compaction: %d near items, %d far items\n", num_near_items, num_far_items);
				//if(step>6) {
				//	printf("\tBefore split far_queue contains: ");
				//	far_frontier->display_items();
				//	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
				//	printf("dist[90145]=%d\n", h_dist[90145]);
				//}
			} while(num_near_items==0);
			thrust::exclusive_scan(thrust::device, valid_near, valid_near + far_size + 1, valid_near);
			thrust::exclusive_scan(thrust::device, valid, valid + far_size + 1, valid);
			in_frontier->reset(); // hold the near items
			near_frontier->reset(); // hold the far items
			//if(step>6) {
			//	printf("\tBefore split far_queue contains: ");
			//	far_frontier->display_items();
			//}
			//compact the far_queue and push some near items into the in_queue
			compact_queue_combo<<<nblocks, nthreads>>>(valid_near, valid, NULL, 0, in_frontier->d_queue, NULL, 0, near_frontier->d_queue, NULL, *far_frontier);
			in_frontier->set_index(num_near_items);
			near_frontier->set_index(num_far_items);
			//printf("\tAfter split: near_size=%d, far_size=%d\n", num_near_items, num_far_items);
			Worklist2 *tmp = far_frontier;
			far_frontier = near_frontier;
			near_frontier = tmp;
			near_frontier->reset();
			//*/
			/*
			//swap the queues
			Worklist2 *tmp = in_frontier;
			in_frontier = far_frontier;
			far_frontier = tmp;
			far_frontier->reset();
			//*/
			in_size = in_frontier->nitems();
			near_size = near_frontier->nitems();
			far_size = far_frontier->nitems();
		} else in_size = 0;
	} while (in_size > 0);

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
