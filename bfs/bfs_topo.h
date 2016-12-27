#define BFS_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
typedef unsigned DistT;

__global__ void initialize(int m, DistT *dist, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		//dist[id] = MYINFINITY;
		expanded[id] = false;
		if(id == 0) visited[id] = true;
		else visited[id] = false;
	}
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, bool *changed, bool *visited, bool *expanded, int *num_frontier) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m && visited[src] && !expanded[src]) { // visited but not expanded
			expanded[src] = true;
			atomicAdd(num_frontier, 1);
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				DistT altdist = dist[src] + 1;
				if (altdist < dist[dst]) {
					DistT olddist = atomicMin(&dist[dst], altdist);
					if (altdist < olddist) {
						*changed = true;
					}
				}
			}
		}
	}
}

__global__ void bfs_update(int m, DistT *dist, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(dist[id] < MYINFINITY && !visited[id])
			visited[id] = true;
	}
}

void BFSSolver(int m, int nnz, int *d_row_offsets, int *d_column_indices, DistT *d_dist) {
	DistT zero = 0;
	bool *d_changed, h_changed, *d_visited, *d_expanded;
	int *d_num_frontier, h_num_frontier;
	double starttime, endtime, runtime;
	int iter = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_frontier, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_expanded, 0, m * sizeof(bool)));
	initialize <<<nblocks, nthreads>>> (m, d_dist, d_visited, d_expanded);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	h_num_frontier = 1;

	const size_t max_blocks = maximum_residency(bfs_kernel, nthreads, 0);
	//const size_t max_blocks = 6;
	//if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, d_changed, d_visited, d_expanded, d_num_frontier);
		bfs_update <<<nblocks, nthreads>>> (m, d_dist, d_visited);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&h_num_frontier, d_num_frontier, sizeof(int), cudaMemcpyDeviceToHost));
		printf("iteration=%d, num_frontier=%d\n", iter, h_num_frontier);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
}
