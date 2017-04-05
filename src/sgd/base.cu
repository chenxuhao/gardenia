// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>

/*
Gardenia Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen
*/
#define SGD_VARIANT "base"
#include "sgd.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void calculate_delta(int num_users, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, LatentT *res_user_lv, LatentT *res_item_lv, ScoreT *total_error) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < num_users) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++) {
				estimate += user_lv[src*K+i] * item_lv[dst*K+i];
			}
			ScoreT delta = rating[offset] - estimate;
			atomicAdd(total_error, fabs(delta));
			for (int i = 0; i < K; i++) {
				//res_user_lv[src*K+i] += user_lv[src*K+i] * delta;
				//res_item_lv[dst*K+i] += item_lv[dst*K+i] * delta;
				LatentT p_s = user_lv[src*K+i];
				LatentT p_d = item_lv[dst*K+i];
				user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
				item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
			}
		}
	}
}

__global__ void update_lv(int m, LatentT *lv, LatentT *res_lv) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		for (int i = 0; i < K; i++) {
			lv[src*K+i] += step * (-lambda * lv[src*K+i] + res_lv[i]);
		}
	}
}

void SGDSolver(int num_users, int num_items, int nnz, int *h_row_offsets, int *h_column_indices, ScoreT *h_rating, LatentT *h_user_lv, LatentT *h_item_lv) {
	print_device_info(0);
	Timer t;
	int iter = 0;

	int *d_row_offsets, *d_column_indices;
	ScoreT *d_rating;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (num_users + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_rating, nnz * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (num_users + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_rating, h_rating, nnz * sizeof(ScoreT), cudaMemcpyHostToDevice));

	LatentT *d_user_lv, *d_item_lv, *res_user_lv, *res_item_lv;
	ScoreT h_error, *d_error;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_user_lv, num_users * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_item_lv, num_items * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&res_user_lv, num_users * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&res_item_lv, num_items * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_user_lv, h_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_item_lv, h_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(ScoreT)));
	//print_latent_vector(num_users, num_items, h_user_lv, h_item_lv);

	int nthreads = 256;
	int nblocks = (num_users - 1) / nthreads + 1;
	//int max_blocks = maximum_residency(calculate_delta, nthreads, 0);
	printf("Solving, nblocks=%d, nthreads=%d\n", nblocks, nthreads);
	t.Start();
	do {
		++iter;
		h_error = 0.0;
		CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(ScoreT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemset(res_user_lv, 0, num_users * K * sizeof(LatentT)));
		CUDA_SAFE_CALL(cudaMemset(res_item_lv, 0, num_items * K * sizeof(LatentT)));
		calculate_delta<<<nblocks, nthreads>>>(num_users, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, res_user_lv, res_item_lv, d_error);
		CudaTest("solving kernel calculate_delta failed");
		//update_lv<<<nblocks, nthreads>>>(num_users, d_user_lv, res_item_lv);
		//update_lv<<<nblocks, nthreads>>>(num_items, d_item_lv, res_user_lv);
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		//print_latent_vector(num_users, num_items, h_user_lv, h_item_lv);
		printf("iteration=%d, error=%f\n", iter, h_error);
	} while (iter < max_iters || h_error < epsilon);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
/*
	ScoreT err = 0.0;
	ScoreT *h_sqerr = (ScoreT *)malloc((num_users + num_items) * sizeof(ScoreT));
	ScoreT *d_sqerr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sqerr, (num_users+num_items) * sizeof(LatentT)));
	for (int i = 0; i < num_users+num_items; i++) {
		h_sqerr[i] = 0.0;
	for (int i = 0; i < num_users; i++) err += h_sqerr[i];
	printf("RMSE error = %lf per edge \n", sqrt(err/nnz));
*/
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_rating));
	CUDA_SAFE_CALL(cudaFree(d_user_lv));
	CUDA_SAFE_CALL(cudaFree(d_item_lv));
	CUDA_SAFE_CALL(cudaFree(res_user_lv));
	CUDA_SAFE_CALL(cudaFree(res_item_lv));
	CUDA_SAFE_CALL(cudaFree(d_error));
}

