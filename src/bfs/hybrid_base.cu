// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "bfs.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
//#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#define BFS_VARIANT "hybrid_base"

__global__ void bottom_up_kernel(int m, const IndexT *row_offsets, const IndexT *column_indices, const int *front, int *next, DistT *depths, int depth) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if(dst < m && depths[dst] == MYINFINITY) { // not visited
		int row_begin = row_offsets[dst];
		int row_end = row_offsets[dst+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int src = column_indices[offset];
			if(__ldg(front+src) == 1) { // if the parent is in the current frontier
				depths[dst] = depth;
				next[dst] = 1; // put this vertex into the next frontier
				break;
			}
		}
	}
}

__global__ void top_down_kernel(int m, const IndexT *row_offsets, const IndexT *column_indices, int *degrees, DistT *depths, int *scout_count, Worklist2 in_queue, Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == MYINFINITY) && (atomicCAS(&depths[dst], MYINFINITY, depths[src]+1)==MYINFINITY)) {
				assert(out_queue.push(dst));
				atomicAdd(scout_count, __ldg(degrees+dst));
			}
		}
	}
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

__global__ void QueueToBitmap(int num, Worklist2 queue, int *bm) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < num) {
		int src;
		if(queue.pop_id(tid, src)) bm[src] = 1;
	}
}

__global__ void BitmapToQueue(int m, int *bm, Worklist2 queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m && bm[tid]) queue.push(tid);
}

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *in_degree, int *h_degree, DistT *h_depths) {
	//print_device_info(0);
	DistT zero = 0;
	int *d_in_row_offsets, *d_in_column_indices;
	int *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	int *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	DistT * d_depths;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	int *d_scout_count;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
	int *front, *next;
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
	
	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	Worklist2 queue1(m), queue2(m);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int alpha = 15, beta = 18;
	int nitems = 1;
	int edges_to_check = nnz;
	int scout_count = h_degree[source];
	insert<<<1, nthreads>>>(source, *in_frontier);

	Timer t;
	t.Start();
	do {
		if(scout_count > edges_to_check / alpha) {
			int awake_count, old_awake_count;
			QueueToBitmap<<<((nitems-1)/512+1), 512>>>(nitems, *in_frontier, front);
			awake_count = nitems;
			do {
				++ iter;
				old_awake_count = awake_count;
				bottom_up_kernel <<<nblocks, nthreads>>> (m, d_in_row_offsets, d_in_column_indices, front, next, d_depths, iter);
				CudaTest("solving bottom_up failed");
				awake_count = thrust::reduce(thrust::device, next, next + m, 0, thrust::plus<int>());
				//printf("BU: (awake_count=%d) ", awake_count);
				//printf("BU: iteration=%d, num_frontier=%d\n", iter, awake_count);
				// swap the queues
				int *temp = front;
				front = next;
				next = temp;
				//thrust::fill(thrust::device, next, next + m, 0);
				CUDA_SAFE_CALL(cudaMemset(next, 0, m * sizeof(int)));
			} while((awake_count >= old_awake_count) || (awake_count > m / beta));
			in_frontier->reset();
			BitmapToQueue<<<((m-1)/512+1), 512>>>(m, front, *in_frontier);
			scout_count = 1;
		} else {
			++ iter;
			edges_to_check -= scout_count;
			nitems = in_frontier->nitems();
			int mblocks = (nitems - 1) / nthreads + 1;
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			top_down_kernel <<<mblocks, nthreads>>> (m, d_out_row_offsets, d_out_column_indices, d_degree, d_depths, d_scout_count, *in_frontier, *out_frontier);
			CudaTest("solving top_down failed");
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			nitems = out_frontier->nitems();
			Worklist2 *tmp = in_frontier;
			in_frontier = out_frontier;
			out_frontier = tmp;
			out_frontier->reset();
			//printf("TD: (scout_count=%d) ", scout_count);
			//printf("TD: iteration=%d, num_frontier=%d\n", iter, nitems);
		}
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_scout_count));
	return;
}
