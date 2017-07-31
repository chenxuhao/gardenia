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
	//__shared__ ScoreT sdata[BLOCK_SIZE/WARP_SIZE*K];                // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int index = warp_id; index < m; index += num_warps) {
		int user_id = ordering[index];
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[user_id + thread_lane];
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[row+1];
		//for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
		for(int offset = row_begin; offset < row_end; offset ++) {
			int item_id = column_indices[offset];
			int base_p = user_id * K;
			int base_q = item_id * K;
			LatentT temp_p[K/WARP_SIZE];
			LatentT temp_q[K/WARP_SIZE];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i += WARP_SIZE) {
				int j = i/WARP_SIZE;
				temp_p[j] = user_lv[base_p+thread_lane+i];
				temp_q[j] = item_lv[base_q+thread_lane+i];
				estimate += temp_p[j] * temp_q[j];
			}
			estimate += __shfl_down(estimate, 16);
			estimate += __shfl_down(estimate, 8);
			estimate += __shfl_down(estimate, 4);
			estimate += __shfl_down(estimate, 2);
			estimate += __shfl_down(estimate, 1);
			estimate = __shfl(estimate, 0);
			ScoreT delta = rating[offset] - estimate;
			for (int i = 0; i < K; i += WARP_SIZE) {
				int j = i/WARP_SIZE;
				LatentT new_user_feature = temp_p[j] + step * (-lambda * temp_p[j] + temp_q[j] * delta);
				LatentT new_item_feature = temp_q[j] + step * (-lambda * temp_q[j] + temp_p[j] * delta);
				user_lv[base_p+thread_lane+i] = new_user_feature;
				item_lv[base_q+thread_lane+i] = new_item_feature;
			}
		}
	}
}

__global__ void compute_rmse(int m, int n, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT *total_error) {
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ ScoreT local_errors[BLOCK_SIZE/WARP_SIZE];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int user_id = warp_id; user_id < m; user_id += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = row_offsets[user_id + thread_lane];
		local_errors[warp_lane] = 0; __syncthreads();
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[row+1];
		//for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
		for(int offset = row_begin; offset < row_end; offset ++) {
			int item_id = column_indices[offset];
			int base_p = user_id * K;
			int base_q = item_id * K;
			ScoreT estimate = 0;
			for (int i = 0; i < K; i += WARP_SIZE) {
				estimate += user_lv[base_p+thread_lane+i] * item_lv[base_q+thread_lane+i];
			}
			estimate += __shfl_down(estimate, 16);
			estimate += __shfl_down(estimate, 8);
			estimate += __shfl_down(estimate, 4);
			estimate += __shfl_down(estimate, 2);
			estimate += __shfl_down(estimate, 1);
			estimate = __shfl(estimate, 0);
			ScoreT error = rating[offset] - estimate;
			if(thread_lane == 0) local_errors[warp_lane] += error*error;
		}
		if(thread_lane == 0) atomicAdd(total_error, local_errors[warp_lane]);
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
	int nblocks = (num_users - 1) / WARPS_PER_BLOCK + 1;
	printf("Launching CUDA SGD solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	compute_rmse<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, d_error);
	CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	printf("iteration %d: RMSE error = %f per edge\n", iter, sqrt(h_error/nnz));

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

