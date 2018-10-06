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

__global__ void calculate_delta(int m, int n, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step, int *ordering) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		int user_id = ordering[tid];
		int row_begin = row_offsets[user_id];
		int row_end = row_offsets[user_id+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int item_id = column_indices[offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++) {
				estimate += user_lv[user_id*K+i] * item_lv[item_id*K+i];
			}
			ScoreT delta = rating[offset] - estimate;
			for (int i = 0; i < K; i++) {
				LatentT p_s = user_lv[user_id*K+i];
				LatentT p_d = item_lv[item_id*K+i];
				LatentT new_user_feature = p_s + step * (-lambda * p_s + p_d * delta);
				LatentT new_item_feature = p_d + step * (-lambda * p_d + p_s * delta);
				user_lv[user_id*K+i] = new_user_feature;
				item_lv[item_id*K+i] = new_item_feature;
			}
		}
	}
}

__global__ void compute_rmse(int m, int n, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT *total_error) {
	int user_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(user_id < m) {
		int row_begin = row_offsets[user_id];
		int row_end = row_offsets[user_id+1];
		ScoreT local_errors = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int item_id = column_indices[offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++) {
				estimate += user_lv[user_id*K+i] * item_lv[item_id*K+i];
			}
			ScoreT error = rating[offset] - estimate;
			local_errors += error*error;
		}
		atomicAdd(total_error, local_errors);
	}
}

void SGDSolver(int num_users, int num_items, int nnz, int *h_row_offsets, int *h_column_indices, ScoreT *h_rating, LatentT *h_user_lv, LatentT *h_item_lv, ScoreT lambda, ScoreT step, int *h_ordering, int max_iters, float epsilon) {
	//print_device_info(0);
	Timer t;
	int iter = 0;
	int *d_row_offsets, *d_column_indices, *d_ordering;
	ScoreT *d_rating;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (num_users + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_rating, nnz * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_ordering, num_users * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (num_users + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_rating, h_rating, nnz * sizeof(ScoreT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_ordering, h_ordering, num_users * sizeof(int), cudaMemcpyHostToDevice));

	LatentT *d_user_lv, *d_item_lv;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_user_lv, num_users * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_item_lv, num_items * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_user_lv, h_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_item_lv, h_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	ScoreT h_error, *d_error;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemset(d_error, 0, sizeof(ScoreT)));

	int nthreads = BLOCK_SIZE;
	int nblocks = (num_users - 1) / nthreads + 1;
	compute_rmse<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, d_error);
	CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	printf("iteration %d: RMSE error = %f per edge\n", iter, sqrt(h_error/nnz));
	//printf("Solving, nblocks=%d, nthreads=%d\n", nblocks, nthreads);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	t.Start();
	do {
		++iter;
		h_error = 0.0;
		CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(ScoreT), cudaMemcpyHostToDevice));
		calculate_delta<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, lambda, step, d_ordering);
		CudaTest("solving kernel calculate_delta failed");
		compute_rmse<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, d_error);
		CudaTest("solving kernel compute_rmse failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
		//printf("h_error=%f\n", h_error);
		assert(h_error>0);
		printf("iteration %d: RMSE error = %f per edge\n", iter, sqrt(h_error/nnz));
		//CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		//print_latent_vector(num_users, num_items, h_user_lv, h_item_lv);
	} while (iter < max_iters && h_error > epsilon);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_rating));
	CUDA_SAFE_CALL(cudaFree(d_user_lv));
	CUDA_SAFE_CALL(cudaFree(d_item_lv));
	CUDA_SAFE_CALL(cudaFree(d_error));
}

