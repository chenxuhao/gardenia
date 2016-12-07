// Copyright (c) 2016, Xuhao Chen

#define CC_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

/*
Gardenia Benchmark Suite
Kernel: Connected Components (CC)
Author: Xuhao Chen

Will return comp array labelling each vertex with a connected component ID
This CC implementation makes use of the Shiloach-Vishkin algorithm
*/

using namespace std;
#define K 20 // dimension of the latent vector (number of features)
#define lambda 0.001
#define step 0.00000035
unsigned int rseed[16*MAX_THREADS];

__global__ void initialize(int m, double *sqerr, double *lv[K]) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		sqerr[id] = 0.0;
		unsigned int r = id;
		for (int j = 0; j < k; j++) {
			lv[id][j] = ((double)rand_r(&r)/(double)RAND_MAX);
		}
	}
}

__global__ void sgd_process(int num_users, int *row_offsets, int *column_indices, int *rating, double *lv[K], double *res_lv[K]) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < num_users) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1]; 
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				double estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += lv[dst][i] * lv[src][i];
				}
				double error = rating[offset] - estimate;
				for (int i =0; i < K; i++) {
					res_lv[i] += lv[dst][i] * error;
				}
			}
		}
	}
}

__global__ void sgd_apply(int num_users, int *row_offsets, int *column_indices, double *lv[K], double *res_lv[K]) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < num_users) {
			for (int i =0; i < K; i++) {
				lv[src][i] += step * (-lambda * lv[src][i] + res_lv[i]);
			}
		}
	}
}

void changed(int m, double *lv_pre[K], double *lv_cur[K], bool * changed) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			for (int i = 0; i < K; i++) {
				if (fabs(lv_pre[src][i] - lv_cur[src][i]) > 1e-7) {
					*changed = true;
				}
			}
		}
	}
}

void SGD(int m, int num_users, int nnz, int *row_offsets, int *column_indices, int *rating, int *degree) {
	const int k = 20;
	srand(0);
	for (int i = 0; i < nthreads; i++) {
		rseed[16*i] = rand();
	}
	double err = 0.0;
	double *h_sqerr, *d_sqerr;
	double starttime, endtime, runtime;
	int iteration = 0;
	h_sqerr = (double *)malloc(m * sizeof(double));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sqerr, sizeof(double) * m));

	double *d_lv[K];
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_lv, sizeof(double) * m * K));
	//CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, sizeof(unsigned) * m, cudaMemcpyHostToDevice));
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	const size_t max_blocks = maximum_residency(sgd_process, nthreads, 0);
	initialize <<<nblocks, nthreads>>> (m, d_lv);
	printf("RMSE error = %lf per edge \n", sqrt(err/nnz));
	//if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iteration;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
		printf("iteration=%d\n", iteration);
		sgd_process<<<nblocks, nthreads>>>(m, row_offsets, column_indices, d_comp, d_changed);
		CudaTest("solving kernel1 failed");
		sgd_apply<<<nblocks, nthreads>>>(m, row_offsets, column_indices, d_comp);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(unsigned) * m, cudaMemcpyDeviceToHost));
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(int m, int *row_offsets, int *column_indices, size_t test_total) {
	size_t total = 0;
	/*
	vector<NodeID> intersection;
	intersection.reserve(m);
	for (NodeID u : g.vertices()) {
		for (NodeID v : g.out_neigh(u)) {
			auto new_end = set_intersection(g.out_neigh(u).begin(),
					g.out_neigh(u).end(),
					g.out_neigh(v).begin(),
					g.out_neigh(v).end(),
					intersection.begin());
			intersection.resize(new_end - intersection.begin());
			total += intersection.size();
		}
	}
	*/
	total = total / 6;  // each triangle was counted 6 times
	if (total != test_total)
		cout << total << " != " << test_total << endl;
	return total == test_total;
}

