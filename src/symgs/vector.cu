// Copyright 2020
// Author: Xuhao Chen <cxh@mit.edu>
#include <stdio.h>
#include "symgs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

template <int VECTORS_PER_BLOCK, int THREADS_PER_VECTOR>
__global__ void gs_kernel(int m, uint64_t * Ap, int * Aj, 
                          int* indices, ValueT * Ax, 
                          ValueT * x, ValueT * b) {
	__shared__ ValueT sdiags[VECTORS_PER_BLOCK];
	__shared__ ValueT sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
	__shared__ IndexT ptrs[VECTORS_PER_BLOCK][2];

	const IndexT THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

	const IndexT thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
	const IndexT thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
	const IndexT vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
	const IndexT vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
	const IndexT num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

	for(IndexT index = vector_id; index < m; index += num_vectors)
	{
		if(thread_lane == 0) sdiags[vector_lane] = 0; __syncthreads();
		IndexT row = indices[index];

		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

		const IndexT row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
		const IndexT row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

		// initialize local sum
		ValueT sum = 0;
///*
		if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
			// ensure aligned memory access to Aj and Ax
			IndexT jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

			// accumulate local sums
			if(jj >= row_start && jj < row_end) {
				IndexT col = Aj[jj];
				bool diag = row == col;
				sum += diag ? 0 : Ax[jj] * x[col];
				if(diag) sdiags[vector_lane] = Ax[jj];
			}

			// accumulate local sums
			for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) {
				IndexT col = Aj[jj];
				bool diag = row == col;
				sum += diag ? 0 : Ax[jj] * x[col];
				if(diag) sdiags[vector_lane] = Ax[jj];
			}
		}
		else {
//*/
			// accumulate local sums
			for(IndexT jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) {
				IndexT col = Aj[jj];
				bool diag = row == col;
				sum += diag ? 0 : Ax[jj] * x[col];
				if(diag) sdiags[vector_lane] = Ax[jj];
			}
		}

		// store local sum in shared memory
		sdata[threadIdx.x] = sum; __syncthreads();

		// reduce local sums to row sum
		if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes the result
		if (thread_lane == 0 && sdiags[vector_lane] != 0)
			x[row] = (b[row] - sdata[threadIdx.x]) / sdiags[vector_lane];
	}
}
size_t nSM;
template <int THREADS_PER_VECTOR>
void gs_gpu(uint64_t *d_Ap, int *d_Aj, 
            int *d_indices, ValueT *d_Ax, 
            ValueT *d_x, ValueT *d_b, 
            int row_start, int row_stop) {
	int m = row_stop - row_start;
	const int VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	//const size_t max_blocks_per_SM = maximum_residency(gs_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, 0);
	//const size_t max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(MAX_BLOCKS, DIVIDE_INTO(m, VECTORS_PER_BLOCK));
	//printf("m=%d, nblocks=%d, nthreads=%d, vector_size=%d\n", m, nblocks, BLOCK_SIZE, THREADS_PER_VECTOR);
	gs_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<nblocks, BLOCK_SIZE>>>(m, d_Ap, d_Aj, d_indices+row_start, d_Ax, d_x, d_b);
}

void gauss_seidel(int m, int nnz, uint64_t *d_Ap, int *d_Aj, 
                  int *d_indices, ValueT *d_Ax, ValueT *d_x, 
                  ValueT *d_b, int row_start, int row_stop, int row_step) {
	int nnz_per_row = nnz / m;
	if (nnz_per_row <=  2) gs_gpu<2>(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, row_start, row_stop);
	else if (nnz_per_row <=  4) gs_gpu<4>(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, row_start, row_stop);
	else if (nnz_per_row <=  8) gs_gpu<8>(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, row_start, row_stop);
	else if (nnz_per_row <= 16) gs_gpu<16>(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, row_start, row_stop);
	else gs_gpu<32>(d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, row_start, row_stop);
}

void SymGSSolver(Graph &g, int *h_indices, 
                 ValueT *h_Ax, ValueT *h_x, 
                 ValueT *h_b, std::vector<int> color_offsets) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_Ap = g.in_rowptr();
  auto h_Aj = g.in_colidx();	
  //print_device_info(0);
  uint64_t *d_Ap;
  VertexId *d_Aj;
	int *d_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_indices, m * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_indices, h_indices, m * sizeof(int), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_b;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m* sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, m* sizeof(ValueT), cudaMemcpyHostToDevice));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	nSM = deviceProp.multiProcessorCount;
	printf("Launching CUDA SymGS solver (%d threads/CTA) ...\n", BLOCK_SIZE);

	Timer t;
	t.Start();
	//printf("Forward\n");
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gauss_seidel(m, nnz, d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i], color_offsets[i+1], 1);
	//printf("Backward\n");
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gauss_seidel(m, nnz, d_Ap, d_Aj, d_indices, d_Ax, d_x, d_b, color_offsets[i-1], color_offsets[i], 1);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [cuda_vector] = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_indices));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_b));
}

