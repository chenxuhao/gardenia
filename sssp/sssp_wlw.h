#define BFS_VARIANT "worklistw"
#define MAXBLOCKSIZE 1024
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void initialize(unsigned *dist, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__global__ void sssp_kernel(int m, int *row_offsets, int *column_indices, W_TYPE *weight, unsigned *dist, Worklist2 inwl, Worklist2 outwl) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int nitems = inwl.nitems();
	//int total_inputs = (nitems - 1) / (gridDim.x * blockDim.x) + 1;
	//for (int id = tid; total_inputs > 0; id += blockDim.x * gridDim.x, total_inputs--) {
		int src;
		if(inwl.pop_id(tid, src)) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				unsigned wt = (unsigned)weight[offset];
				unsigned altdist = dist[src] + wt;
				if (altdist < dist[dst]) {
					unsigned olddist = atomicMin(&dist[dst], altdist);
					if (altdist < olddist) { // update successfully
						assert(outwl.push(dst));
					}
				}
			}
		}
	//}
}

__global__ void insert(Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(0);
	}
	return;
}

void sssp(int m, int nnz, int *d_row_offsets, int *d_column_indices, W_TYPE *d_weight, unsigned *d_dist) {
	unsigned zero = 0;
	int iteration = 0;
	double starttime, endtime, runtime;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (d_dist, m);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	unsigned nitems = 1;
	//const size_t max_blocks = maximum_residency(sssp_kernel, nthreads, 0);
	starttime = rtclock();
	insert<<<1, nthreads>>>(*inwl);
	nitems = inwl->nitems();
	do {
		++iteration;
		nblocks = (nitems - 1) / nthreads + 1;
		//printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iteration, nblocks, nthreads, nitems);
		sssp_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, d_dist, *inwl, *outwl);
		CudaTest("solving failed");
		nitems = outwl->nitems();
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	return;
}
