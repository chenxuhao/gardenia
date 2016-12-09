// Copyright (c) 2016, Xuhao Chen
/*
Gardenia Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen
*/
#define CC_VARIANT "naive"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define K 20 // dimension of the latent vector (number of features)
#define lambda 0.001
#define step 0.00000035
#define epsilon 1e-3
#define max_iters 19
/*
__global__ void initialize(int m, double *sqerr, double *lv[K]) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		sqerr[id] = 0.0;
		unsigned int r = id;
		for (int j = 0; j < K; j++) {
			lv[id][j] = ((double)rand_r(&r)/(double)RAND_MAX);
		}
	}
}
*/
__global__ void sgd_process(int num_users, int *row_offsets, int *column_indices, W_TYPE *rating, double *lv, double *res_lv, double *total_error) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (num_users - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < num_users) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1]; 
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				double estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += lv[dst*K+i] * lv[src*K+i];
				}
				double error = rating[offset] - estimate;
				*total_error += fabs(error);
				for (int i =0; i < K; i++) {
					res_lv[i] += lv[dst*K+i] * error;
				}
			}
		}
	}
}

__global__ void sgd_apply(int num_users, int *row_offsets, int *column_indices, double *lv, double *res_lv) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (num_users - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < num_users) {
			for (int i =0; i < K; i++) {
				lv[src*K+i] += step * (-lambda * lv[src*K+i] + res_lv[i]);
			}
		}
	}
}
/*
__global__ void changed(int m, double *lv_pre, double *lv_cur, bool * changed) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			for (int i = 0; i < K; i++) {
				if (fabs(lv_pre[src*K+i] - lv_cur[src*K+i]) > 1e-7) {
					*changed = true;
				}
			}
		}
	}
}
*/

void sgd(int m, int num_users, int nnz, int *row_offsets, int *column_indices, W_TYPE *rating) {
	double starttime, endtime, runtime;
	double h_error, *d_error;
	const int k = 20;
	srand(0);
	double err = 0.0;
	double *h_sqerr, *d_sqerr;
	double *h_lv, *d_lv, *res_lv;
	int iter = 0;
	h_sqerr = (double *)malloc(m * sizeof(double));
	h_lv = (double *)malloc(m * K * sizeof(double));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sqerr, sizeof(double) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_lv, sizeof(double) * m * K));
	CUDA_SAFE_CALL(cudaMalloc((void **)&res_lv, sizeof(double) * m * K));
	for (int i = 0; i < m; i++) {
		h_sqerr[i] = 0.0;
		unsigned int r = i;
		for (int j = 0; j < K; j++) {
			h_lv[i*K+j] = ((double)rand_r(&r)/(double)RAND_MAX);
		} 
	}
	for (int i = 0; i < m; i++) err += h_sqerr[i];
	CUDA_SAFE_CALL(cudaMemcpy(d_sqerr, &h_sqerr, sizeof(double) * m, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_lv, &h_lv, sizeof(double) * m * K, cudaMemcpyHostToDevice));
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	const size_t max_blocks = maximum_residency(sgd_process, nthreads, 0);
	//initialize <<<nblocks, nthreads>>> (m, d_sqerr, d_lv);
	printf("RMSE error = %lf per edge \n", sqrt(err/nnz));
	//if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iter;
		h_error = 0.0;
		CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(double), cudaMemcpyHostToDevice));
		printf("iteration=%d\n", iter);
		sgd_process<<<nblocks, nthreads>>>(num_users, row_offsets, column_indices, rating, d_lv, res_lv, d_error);
		CudaTest("solving kernel1 failed");
		sgd_apply<<<nblocks, nthreads>>>(m, row_offsets, column_indices, d_lv, res_lv);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(h_error), cudaMemcpyDeviceToHost));
	} while (iter < max_iters || h_error < epsilon);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_lv, d_lv, sizeof(unsigned) * m * K, cudaMemcpyDeviceToHost));
	endtime = rtclock();
	printf("\titerations = %d.\n", iter);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_error));
}

// Compares with simple serial implementation that uses std::set_intersection
bool SGDVerifier(int m, int *row_offsets, int *column_indices, size_t test_total) {
	size_t total = 0;
	total = total / 6;  // each triangle was counted 6 times
	if (total != test_total)
		cout << total << " != " << test_total << endl;
	return total == test_total;
}

