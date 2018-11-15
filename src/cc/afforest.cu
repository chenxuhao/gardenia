// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define CC_VARIANT "afforest"

__device__ void link(IndexT u, IndexT v, IndexT *comp) {
	IndexT p1 = comp[u];
	IndexT p2 = comp[v];
	while (p1 != p2) {
		IndexT high = p1 > p2 ? p1 : p2;
		IndexT low = p1 + (p2 - high);
		IndexT p_high = comp[high];
		if ((p_high == low) || (p_high == high && atomicCAS(&comp[high], high, low) == high))
			break;
		p1 = comp[comp[high]];
		p2 = comp[low];
	}
}

__global__ void compress(int m, CompT *comp) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

__global__ void afforest(int m, const IndexT *row_offsets, const IndexT *column_indices, CompT *comp, int32_t r) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		int start_offset = min(r, row_end - row_begin);
		row_begin += start_offset;
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			link(src, dst, comp);
			break;
		}
	}
}

__global__ void afforest_undirected(int m, int c, const IndexT *row_offsets, const IndexT *column_indices, CompT *comp, int r) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && comp[src] != c) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		int start_offset = min(r, row_end - row_begin);
		row_begin += start_offset;
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			link(src, dst, comp);
		}
	}
}

__global__ void afforest_directed(int m, int c, const IndexT *in_row_offsets, const IndexT *in_column_indices, const IndexT *row_offsets, const IndexT *column_indices, CompT *comp, int r) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m && comp[src] != c) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		int start_offset = min(r, row_end - row_begin);
		row_begin += start_offset;
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = column_indices[offset];
			link(src, dst, comp);
		}
		row_begin = in_row_offsets[src];
		row_end = in_row_offsets[src+1];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT dst = in_column_indices[offset];
			link(src, dst, comp);
		}
	}
}

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, CompT *h_comp, bool is_directed) {
	//print_device_info(0);
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
	CompT *d_comp;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, m * sizeof(CompT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, m * sizeof(CompT), cudaMemcpyHostToDevice));

	int neighbor_rounds = 2;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA CC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	for (int r = 0; r < neighbor_rounds; ++r) {
		afforest<<<nblocks, nthreads>>>(m, d_out_row_offsets, d_out_column_indices, d_comp, r);
		CudaTest("solving kernel afforest failed");
		compress<<<nblocks, nthreads>>>(m, d_comp);
		CudaTest("solving kernel compress failed");
	}
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, m * sizeof(CompT), cudaMemcpyDeviceToHost));
	IndexT c = SampleFrequentElement(m, h_comp);
	if (!is_directed) {
		afforest_undirected<<<nblocks, nthreads>>>(m, c, d_out_row_offsets, d_out_column_indices, d_comp, neighbor_rounds);
	} else {
		afforest_directed<<<nblocks, nthreads>>>(m, c, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_comp, neighbor_rounds);
	}
	compress<<<nblocks, nthreads>>>(m, d_comp);
	CudaTest("solving kernel compress failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, m * sizeof(CompT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
}

