// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
// Topology-driven Minimum Spanning Tree using CUDA
#include "common.h"
#include "timer.h"
#include "graph_io.h"
#include "gbar.h"
#include "component.h"
#include "cuda_launch_config.hpp"
#define MST_TYPE unsigned

__global__ void dinit(int m, MST_TYPE *eleminwts, 
                      MST_TYPE *minwtcomponent, unsigned *partners, 
                      bool *processinnextiteration, 
                      unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;
		partners[id] = id;
		goaheadnodeofcomponent[id] = m;
		processinnextiteration[id] = false;
	}
}

__global__ void dfindelemin(int m, int *row_offsets, 
                            int *column_indices, WeightT *weight, 
                            MST_TYPE *mstwt, ComponentSpace cs, 
                            MST_TYPE *eleminwts, MST_TYPE *minwtcomponent, 
                            unsigned *partners) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = m;
		MST_TYPE minwt = MYINFINITY;
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1];
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
			MST_TYPE wt = (MST_TYPE)weight[offset];
			if (wt < minwt) {
				unsigned dst = column_indices[offset];
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		eleminwts[id] = minwt;
		partners[id] = dstboss;
		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			atomicMin(&minwtcomponent[srcboss], minwt);
		}
	}
}

__global__ void dfindelemin2(int m, int *row_offsets, 
                             int *column_indices, WeightT *weight, 
                             ComponentSpace cs, MST_TYPE *eleminwts, 
                             MST_TYPE *minwtcomponent, unsigned *partners, 
                             unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		if(eleminwts[id] == minwtcomponent[srcboss] && 
       srcboss != partners[id] && partners[id] != m) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				MST_TYPE wt = (MST_TYPE)weight[offset];
				if (wt == eleminwts[id]) {
					unsigned dst = column_indices[offset];
					unsigned tempdstboss = cs.find(dst);
					if (tempdstboss == partners[id]) {
						atomicCAS(&goaheadnodeofcomponent[srcboss], m, id);
					}
				}
			}
		}
	}
}

__global__ void verify_min_elem(int m, int *row_offsets, 
                                int *column_indices, WeightT *weight, 
                                ComponentSpace cs, MST_TYPE *minwtcomponent, 
                                unsigned *partners, bool *processinnextiteration, 
                                unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(cs.isBoss(id)) {
			if(goaheadnodeofcomponent[id] == m) {
				return;
			}
			unsigned minwt_node = goaheadnodeofcomponent[id];
			MST_TYPE minwt = minwtcomponent[id];
			if(minwt == MYINFINITY)
				return;
			unsigned row_begin = row_offsets[minwt_node];
			unsigned row_end = row_offsets[minwt_node + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				MST_TYPE wt = (MST_TYPE)weight[offset];
				if (wt == minwt) {
					unsigned dst = column_indices[offset];
					unsigned tempdstboss = cs.find(dst);
					if(tempdstboss == partners[minwt_node] && tempdstboss != id) {
						processinnextiteration[minwt_node] = true;
						return;
					}
				}
			}
		}
	}
}

__device__ volatile int g_mutex;
__device__ void __gpu_sync_atomic(int goalVal) {
	int tid = threadIdx.x * blockDim.y + threadIdx.y;
	__threadfence();
	__syncthreads();
	if (tid == 0) {
		atomicAdd((int *)&g_mutex, 1);
		while(g_mutex % goalVal != 0) {} // Note: this causes GPGPU-Sim not terminating, need to implement 'volatile' in GPGPU-Sim
	}
	__syncthreads();
}

__global__ void dfindcompmintwo(int m, unsigned *mstwt, ComponentSpace csw, 
                                MST_TYPE *eleminwts, MST_TYPE *minwtcomponent, 
                                unsigned *partners, bool *processinnextiteration, 
                                GlobalBarrier gb, bool *repeat, unsigned *count) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id, nthreads = blockDim.x * gridDim.x;
	unsigned up = (m + nthreads - 1) / nthreads * nthreads;
	unsigned srcboss, dstboss;
	for(id = tid; id < up; id += nthreads) {
		if(id < m && processinnextiteration[id]) {
			srcboss = csw.find(id);
			dstboss = csw.find(partners[id]);
		}
		//gb.Sync();
		__gpu_sync_atomic(gridDim.x);
		if (id < m && processinnextiteration[id] && srcboss != dstboss) {
			//printf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);
			if (csw.unify(srcboss, dstboss)) {
				atomicAdd(mstwt, eleminwts[id]);
				atomicAdd(count, 1);
				//printf("u %d -> %d (%d)\n", srcboss, dstboss, eleminwts[id]);
				processinnextiteration[id] = false;
				eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
			}
			else {
				*repeat = true;
			}
			//printf("\tcomp[%d] = %d.\n", srcboss, csw.find(srcboss));
		}
		//gb.Sync();
		__gpu_sync_atomic(gridDim.x);
	}
}

int main(int argc, char *argv[]) {
	printf("Minimum Spanning Tree by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);
	print_device_info(0);
	WeightT *d_weight;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(WeightT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(WeightT), cudaMemcpyHostToDevice));
	int mutex = 0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_mutex, &mutex, sizeof(int)));
	
	MST_TYPE *mstwt, hmstwt = 0;
	int iteration = 0;
	unsigned *partners;
	MST_TYPE *eleminwts, *minwtcomponent;
	bool *processinnextiteration;
	unsigned *goaheadnodeofcomponent;
	ComponentSpace cs(m);
	unsigned prevncomponents, currncomponents = m;
	bool repeat = false, *grepeat;
	unsigned edgecount = 0, *gedgecount;

	CUDA_SAFE_CALL(cudaMalloc((void **)&mstwt, sizeof(MST_TYPE)));
	CUDA_SAFE_CALL(cudaMemcpy(mstwt, &hmstwt, sizeof(MST_TYPE), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&eleminwts, m * sizeof(MST_TYPE)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&minwtcomponent, m * sizeof(MST_TYPE)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&partners, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&processinnextiteration, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&goaheadnodeofcomponent, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc(&grepeat, sizeof(bool) * 1));
	CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&gedgecount, sizeof(unsigned) * 1));
	CUDA_SAFE_CALL(cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	int nSM = 13;
	//const size_t max_blocks = maximum_residency(dfindcompmintwo, nthreads, 0);
	int max_blocks = 1;
	printf("Setup global barrier, max_blocks=%d\n", max_blocks);
	GlobalBarrierLifetime gb;
	gb.Setup(nSM * max_blocks);
	printf("Finding mst...\n");
	Timer t;
	t.Start();
	do {
		++iteration;
		prevncomponents = currncomponents;
		dinit<<<nblocks, nthreads>>>(m, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
		CudaTest("dinit failed");
		dfindelemin<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, mstwt, cs, eleminwts, minwtcomponent, partners);
		dfindelemin2<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, cs, eleminwts, minwtcomponent, partners, goaheadnodeofcomponent);
		verify_min_elem<<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, cs, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
		CudaTest("dfindelemin failed");
		do {
			repeat = false;
			CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
			dfindcompmintwo <<<nSM * max_blocks, nthreads>>> (m, mstwt, cs, eleminwts, minwtcomponent, partners, processinnextiteration, gb, grepeat, gedgecount);
			CudaTest("dfindcompmintwo failed");
			CUDA_SAFE_CALL(cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost));
		} while (repeat); // only required for quicker convergence?
		currncomponents = cs.numberOfComponentsHost();
		CUDA_SAFE_CALL(cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
		printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, hmstwt, edgecount);
	} while (currncomponents != prevncomponents);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
	printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
	printf("\truntime [mst] = %f ms.\n", t.Millisecs());
	return 0;
}
