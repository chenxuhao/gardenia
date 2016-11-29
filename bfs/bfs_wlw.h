#define BFS_VARIANT "vertex-serial"
#define MAXBLOCKSIZE 1024
#include "worklistc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#ifdef TEXTURE
texture <int, 1, cudaReadModeElementType> row_offsets;
texture <int, 1, cudaReadModeElementType> column_indices;
#endif
__global__ void initialize(unsigned *dist, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

#ifdef TEXTURE
__global__ void bfs_kernel(int m, foru *dist, Worklist2 inwl, Worklist2 outwl) {
#else
__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, foru *dist, Worklist2 inwl, Worklist2 outwl) {
#endif
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(inwl.pop_id(tid, src)) {
#ifdef TEXTURE
		unsigned row_begin = tex1Dfetch(row_offsets, src);
		unsigned row_end = tex1Dfetch(row_offsets, src + 1);
#else
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1];
#endif
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
#ifdef TEXTURE
			int dst = tex1Dfetch(column_indices, offset);
#else
			int dst = column_indices[offset];
#endif
			foru altdist = dist[src] + 1;
			if (dist[dst] == MYINFINITY) {//Not visited
				dist[dst] = altdist;
				assert(outwl.push(dst));
			}
		}
	}
}

__global__ void insert(Worklist2 inwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		inwl.push(0);
	}
	return;
}

void bfs(int m, int nnz, int *d_row_offsets, int *d_column_indices, unsigned *d_dist, int nSM) {
	foru zero = 0;
	int iteration = 0;
	double starttime, endtime, runtime;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
#ifdef TEXTURE
	CUDA_SAFE_CALL(cudaBindTexture(0, row_offsets, d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaBindTexture(0, column_indices, d_column_indices, (nnz + 1) * sizeof(int)));
#endif
	//initialize <<<nblocks, nthreads>>> (d_dist, m);
	//CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice));
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	unsigned nitems = 1;
	starttime = rtclock();
	insert<<<1, nthreads>>>(*inwl);
	nitems = inwl->nitems();
	do {
		++iteration;
		nblocks = (nitems - 1) / nthreads + 1;
		printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iteration, nblocks, nthreads, nitems);
#ifdef TEXTURE
		bfs_kernel <<<nblocks, nthreads>>> (m, d_dist, *inwl, *outwl);
#else
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, *inwl, *outwl);
#endif
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
