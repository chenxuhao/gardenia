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
#include <cub/cub.cuh>
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void update(int m, int n, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step, int *ordering, ScoreT *squared_errors) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		//int user_id = ordering[tid];
		int user_id = tid;
		int row_begin = row_offsets[user_id];
		int row_end = row_offsets[user_id+1]; 
		int user_offset = K * user_id;
		LatentT *ulv = &user_lv[user_offset];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int item_id = column_indices[offset];
			int item_offset = K * item_id;
			LatentT *ilv = &item_lv[item_offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++)
				estimate += ulv[i] * ilv[i];
			ScoreT delta = rating[offset] - estimate;
			squared_errors[user_id] += delta * delta;
			for (int i = 0; i < K; i++) {
				LatentT p_u = ulv[i];
				LatentT p_i = ilv[i];
				ulv[i] += step * (-lambda * p_u + p_i * delta);
				ilv[i] += step * (-lambda * p_i + p_u * delta);
			}
		}
	}
}

__global__ void rmse(int m, ScoreT *squared_errors, ScoreT *total_error) {
	int uid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	ScoreT local_error = 0.0;
	if(uid < m) local_error = squared_errors[uid];
	ScoreT block_sum = BlockReduce(temp_storage).Sum(local_error);
	if(threadIdx.x == 0) atomicAdd(total_error, block_sum);
}

void SGDSolver(int num_users, int num_items, int nnz, int *h_row_offsets, int *h_column_indices, ScoreT *h_rating, LatentT *h_user_lv, LatentT *h_item_lv, int *h_ordering) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (num_users + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (num_users + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_rating;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_rating, nnz * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_rating, h_rating, nnz * sizeof(ScoreT), cudaMemcpyHostToDevice));
	int *d_ordering;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_ordering, num_users * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMemcpy(d_ordering, h_ordering, num_users * sizeof(int), cudaMemcpyHostToDevice));

	LatentT *d_user_lv, *d_item_lv;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_user_lv, num_users * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_item_lv, num_items * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_user_lv, h_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_item_lv, h_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	ScoreT h_error, *d_error, *squared_errors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&squared_errors, num_users * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemset(d_error, 0, sizeof(ScoreT)));

	int iter = 0;
	int nthreads = BLOCK_SIZE;
	int nblocks = (num_users - 1) / nthreads + 1;
	printf("Launching CUDA SGD solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Timer t;
	t.Start();
	do {
		++iter;
		h_error = 0.0;
		CUDA_SAFE_CALL(cudaMemset(squared_errors, 0, num_users * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(ScoreT), cudaMemcpyHostToDevice));
		update<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, lambda, step, d_ordering, squared_errors);
		CudaTest("solving kernel update failed");
		rmse<<<nblocks, nthreads>>>(num_users, squared_errors, d_error);
		CudaTest("solving kernel rmse failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
		//printf("h_error=%f\n", h_error);
		printf("iteration %d: RMSE error = %f\n", iter, sqrt(h_error/nnz));
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
	CUDA_SAFE_CALL(cudaFree(squared_errors));
}

