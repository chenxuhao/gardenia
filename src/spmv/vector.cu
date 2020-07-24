// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <algorithm>
#define SPMV_VARIANT "vector"

// CSR SpMV kernels based on a vector model (one vector per row)
//   Each row of the CSR matrix is assigned to a vector.  The vector computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire vector is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a vector.

texture<float,1> tex_x;
void bind_x(const float * x) { CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_x, x)); }
void unbind_x(const float * x) { CUDA_SAFE_CALL(cudaUnbindTexture(tex_x)); }

template <int VECTORS_PER_BLOCK, int THREADS_PER_VECTOR>
__global__ void spmv_vector_kernel(int m, const uint64_t* Ap,
                                   const VertexId * Aj, const ValueT * Ax, 
                                   const ValueT * x, ValueT * y) {
	__shared__ ValueT sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]; // padded to avoid reduction ifs
	__shared__ int ptrs[VECTORS_PER_BLOCK][2];

	const int thread_id	  = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR-1);   // thread index within the vector
	const int vector_id   = thread_id   / THREADS_PER_VECTOR;     // global vector index
	const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;     // vector index within the CTA
	const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;        // total number of active vectors

	for(auto row = vector_id; row < m; row += num_vectors) {
		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
		auto row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
		auto row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

		// compute local sum
		ValueT sum = 0;
		for(auto offset = row_start + thread_lane; offset < row_end; offset += THREADS_PER_VECTOR)
			//sum += Ax[offset] * x[Aj[offset]];
			sum += Ax[offset] * tex1Dfetch(tex_x, Aj[offset]);

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum; __syncthreads();
		if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes vector result
		if (thread_lane == 0)
			y[row] += sdata[threadIdx.x];
	}
}

size_t nSM;
template <int THREADS_PER_VECTOR>
void spmv_vector(int m, uint64_t *d_Ap, VertexId *d_Aj, ValueT *d_Ax, ValueT *d_x, ValueT *d_y) {
	const int VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	//const size_t max_blocks_per_SM = maximum_residency(spmv_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, 0);
	//const size_t max_blocks = max_blocks_per_SM * nSM;
	const int nblocks = std::min(MAX_BLOCKS, DIVIDE_INTO(m, VECTORS_PER_BLOCK));
	spmv_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<nblocks, BLOCK_SIZE>>>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
	CudaTest("solving failed");
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_Ap = g.in_rowptr();
  auto h_Aj = g.in_colidx();	
  //print_device_info(0);
  uint64_t *d_Ap;
  VertexId *d_Aj;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

  ValueT *d_Ax, *d_x, *d_y;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
  CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));

  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  nSM = deviceProp.multiProcessorCount;
  int nnz_per_row = nnz / m;
  printf("Launching CUDA SpMV solver (%d threads/CTA) ...\n", BLOCK_SIZE);

  Timer t;
  t.Start();
  bind_x(d_x);
  if (nnz_per_row <=  2) spmv_vector<2>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  else if (nnz_per_row <=  4) spmv_vector<4>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  else if (nnz_per_row <=  8) spmv_vector<8>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  else if (nnz_per_row <= 16) spmv_vector<16>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  else spmv_vector<32>(m, d_Ap, d_Aj, d_Ax, d_x, d_y);
  unbind_x(d_x);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_Ap));
  CUDA_SAFE_CALL(cudaFree(d_Aj));
  CUDA_SAFE_CALL(cudaFree(d_Ax));
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
}

