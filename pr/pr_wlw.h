#define BFS_VARIANT "worklistw"
#include <cub/cub.cuh>
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"
#define DELTA 0.00000001
#define EPSILON 0.01
#define ITER 10

__global__ void initialize(float *cur_pagerank, float *next_pagerank, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_pagerank[id] = 1.0f / (float)m;
		next_pagerank[id] = 1.0f / (float)m;
	}
}

__global__ void update_neighbors(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank, float *diff, Worklist2 inwl, Worklist2 outwl) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int ntimes = *inwl.dindex;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int id = tid; total_inputs > 0; id += blockDim.x * gridDim.x, total_inputs--) {
		int src = id;
		if (src < m) {
		//if (inwl.pop_id(id, src)) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			unsigned degree = row_end - row_begin;
			float value = 0.85 * cur_pagerank[src] / (float)degree;
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				atomicAdd(&next_pagerank[dst], value);
			}
		}
	}
}

__global__ void self_update(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank, float *diff, Worklist2 inwl, Worklist2 outwl) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockReduce<float, 256> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int id = tid; total_inputs > 0; id += blockDim.x * gridDim.x, total_inputs--) {
		int src = id;
		if (src < m) {
		//if (inwl.pop_id(id, src)) {
			float delta = abs(next_pagerank[src] - cur_pagerank[src]);
			//if (delta > DELTA) outwl.push(src); // push this vertex into the frontier
			local_diff += delta;
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = 0.15 / (float)m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, foru* d_weight, int nSM) {
	unsigned zero = 0;
	float *d_diff, h_diff;
	float *d_cur_pagerank, *d_next_pagerank;
	double starttime, endtime, runtime;
	int iteration = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_cur_pagerank, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_pagerank, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	initialize <<<nblocks, nthreads>>> (d_cur_pagerank, d_next_pagerank, m);
	CudaTest("initializing failed");

	Worklist2 wl1(m), wl2(m); 
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	unsigned nitems = m;
	for(int i = 0; i < m; i ++) {
		inwl->wl[i] = i;
	}
	CUDA_SAFE_CALL(cudaMemcpy(inwl->dindex, &m, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(inwl->dwl, inwl->wl, m * sizeof(int), cudaMemcpyHostToDevice));
	const size_t max_blocks = maximum_residency(update_neighbors, nthreads, 0);
	//const size_t max_blocks = maximum_residency(self_update, nthreads, 0);
	//printf("max_blocks=%d\n", max_blocks);
	//const size_t max_blocks = 5;
	//printf("Solving, nSM=%d, max_blocks=%d, nblocks=%d, nthreads=%d\n", nSM, max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iteration;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		nblocks = (nitems - 1) / nthreads + 1;
		if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
		update_neighbors <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank, d_diff, *inwl, *outwl);
		self_update <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank, d_diff, *inwl, *outwl);
		CudaTest("solving failed");
		//nitems = outwl->nitems();
		//Worklist2 *tmp = inwl;
		//inwl = outwl;
		//outwl = tmp;
		//outwl->reset();
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f, nitems=%d\n", iteration, h_diff, nitems);
	} while (h_diff > EPSILON && iteration < ITER && nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
