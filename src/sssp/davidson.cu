// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#define SSSP_VARIANT "base"
#include "sssp.h"
#include "timer.h"
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define COMPACT_THRESHOLD 50000
#define BUCKET_MIN 0
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
	int iteration = 0;
	Timer t;
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
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	int max_blocks = maximum_residency(bellman_ford, nthreads, 0);
	printf("Launching CUDA SSSP solver (%d CTAs/SM, %d threads/CTA) ...\n", max_blocks, nthreads);
	t.Start();
	insert<<<1, nthreads>>>(source, *inwl);
	nitems = inwl->nitems();

	Buckets<FarPile> m_b(source, 1, delta, nnz, m);
	unsigned int step = 0;	
	float total = 0;
	unsigned int workFront;
	unsigned int *sizeA = (unsigned int*) malloc(sizeof(unsigned int)*1); 
	unsigned int currBucket = 0;
	unsigned int* farBucket = (unsigned int*) malloc(sizeof(unsigned int)); 
	farBucket[0] = 1;
	unsigned int *splitIndex; 
	cudaMalloc((void**) &splitIndex, sizeof(unsigned int));
	cudaMemset(nodeLock, 0xFF, m * sizeof(unsigned int));		
	cudaMemset(minBucket, 0xFF, m * sizeof(unsigned int));			
	int nodesFound = 0;
	int prevNodes = 0;
	int nodesProcessed;
	unsigned int workFrontThisIter = 0;
	float alpha = 0;
	bool farPile = true;
	do {
		step ++;
		if(m_b.zero_elements > 0) {
			workFrontThisIter += m_b.zero_elements;
			relaxVertexFront<CT, WT, FarPile, true, BFSType, NT>
				(d_s, m_b, csr, workFront, sizeA, context, step);
		}
		m_b.zero_elements = 0;
		m_b.num_elements = 0;
		bool compact = true;
		if(compact && workFront > 0) {
			if(FarPile) {
				splitWork<CT, WT, NT, true, FarPile>(d_s, m_b, d_s.gpuOut.vertex, 
						d_s.gpuOut.bucketId, workFront, farBucket[0]-1, farBucket[0], sizeA, farPile, context);	
			}
			else {
				compactSet<CT, WT, NT, false, true>(d_s, d_s.gpuOut.vertex, NULL,
						m_b.gpu.vertex, NULL, d_s.gpuOut.validFlag, workFront, context, 0);									
				cudaMemcpy(sizeA, d_s.gpuOut.validFlag+workFront, 
						sizeof(int), cudaMemcpyDeviceToHost);	
				m_b.zero_elements = m_b.num_elements = sizeA[0];
				//printf("After compacting set %d elements m_b.farSize %d\n", m_b.zero_elements, m_b.farSize);
			}
		}
		else if( workFront > 0) {
			if(FarPile)
				splitWork<CT, WT, NT, true>(d_s, m_b, d_s.gpuOut.vertex, 
						d_s.gpuOut.bucketId, workFront, farBucket[0]-1, farBucket[0], sizeA, farPile, context);		
			else {
				int nB = (workFront + NT - 1)/NT;
				compactFlags<<<nB, NT>>> (d_s.gpuOut.validFlag, d_s.gpuOut.vertex, m_b.gpu.vertex, workFront);	
				cudaMemcpy(sizeA, d_s.gpuOut.validFlag+workFront, sizeof(int), cudaMemcpyDeviceToHost);	
				m_b.zero_elements = m_b.num_elements = sizeA[0];
			}
		}
		if( (m_b.num_elements == 0 && m_b.farSize > 0)) {
			workFront = m_b.farSize;
			farBucket[0]++;			
			//printf("Splitting Far Pile! %d farBucket[0] %d\n", workFront, farBucket[0]);
			if(workFront > 0) {
				if(farPile)
					splitWork<CT, WT, NT, FarPile>
						(d_s, m_b, m_b.gpu.farVertex, m_b.gpu.farBuckets, d_s.gpuOut.vertexTemp, 
						 workFront, farBucket[0]-1, farBucket[0], m_b.gpu.vertex, m_b.gpu.bucketId, 
						 m_b.num_elements, 0, d_s.gpuOut.vertexTemp, 
						 d_s.gpuOut.bucketIdTemp, m_b.farSize, 0, sizeA, context);				
				else
					splitWork<CT, WT, NT, FarPile>
						(d_s, m_b, d_s.gpuOut.vertexTemp, d_s.gpuOut.bucketIdTemp, m_b.gpu.farVertex,
						 workFront, farBucket[0]-1, farBucket[0], m_b.gpu.vertex, m_b.gpu.bucketId,
						 m_b.num_elements, 0, m_b.gpu.farVertex, m_b.gpu.farBuckets, m_b.farSize, 0, sizeA, context);				
				farPile = !farPile;
			}
			else {
				m_b.farSize = 0;
			}
			m_b.zero_elements = m_b.num_elements;
			workFront = 0;
		}
	}
	while(m_b.num_elements != 0 || m_b.farSize != 0);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iteration);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	return;
}
