#pragma once
#include "cutil_subset.h"
static int zero = 0;

struct Worklist {
	int *d_queue, *h_queue;
	int *d_size, *d_index;

	Worklist(size_t max_size) {
		h_queue = (int *) calloc(max_size, sizeof(int));
		CUDA_SAFE_CALL(cudaMalloc(&d_queue, max_size * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_size, sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_index, sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(d_size, &max_size, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &zero, sizeof(zero), cudaMemcpyHostToDevice));
	}

	~Worklist() {}

	void display_items() {
		int nsize = nitems();
		CUDA_SAFE_CALL(cudaMemcpy(h_queue, d_queue, nsize  * sizeof(int), cudaMemcpyDeviceToHost));
		printf("Queue: ");
		for(int i = 0; i < nsize; i++)
			printf("%d %d, ", i, h_queue[i]);
		printf("\n");
		return;
	}

	void reset() {
		CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &zero, sizeof(int), cudaMemcpyHostToDevice));
	}

	int nitems() {
		int index;
		CUDA_SAFE_CALL(cudaMemcpy(&index, (void *)d_index, sizeof(int), cudaMemcpyDeviceToHost));
		return index;
	}

	void set_index(int index) {
		CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &index, sizeof(int), cudaMemcpyHostToDevice));
	}

	__device__ int push(int item) {
		int lindex = atomicAdd((int *) d_index, 1);
		if(lindex >= *d_size)
			return 0;
		d_queue[lindex] = item;
		return 1;
	}

	__device__ int pop(int &item) {
		int lindex = atomicSub((int *) d_index, 1);
		if(lindex <= 0) {
			*d_index = 0;
			return 0;
		}
		item = d_queue[lindex - 1];
		return 1;
	}
};

struct Worklist2: public Worklist {
	Worklist2(int nsize) : Worklist(nsize) {}

	template <typename T> __device__ __forceinline__
		int push_1item(int nitem, int item, int threads_per_block) {
			assert(nitem == 0 || nitem == 1);
			__shared__ typename T::TempStorage temp_storage;
			__shared__ int queue_index;
			int total_items = 0;
			int thread_data = nitem;
			T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);
			__syncthreads();
			if(threadIdx.x == 0) {	
				queue_index = atomicAdd((int *) d_index, total_items);
			}
			__syncthreads();
			if(nitem == 1) {
				if(queue_index + thread_data >= *d_size) {
					printf("GPU: exceeded size: %d %d %d %d %d\n", queue_index, thread_data, *d_size, total_items, *d_index);
					return 0;
				}
				//cub::ThreadStore<cub::STORE_CG>(d_queue + queue_index + thread_data, item);
				d_queue[queue_index + thread_data] = item;
			}
			__syncthreads();
			return total_items;
		}

	template <typename T>
		__device__ __forceinline__
		int push_nitems(int n_items, int *items, int threads_per_block) {
			__shared__ typename T::TempStorage temp_storage;
			__shared__ int queue_index;
			int total_items;
			int thread_data = n_items;
			T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);
			if(threadIdx.x == 0) {	
				queue_index = atomicAdd((int *) d_index, total_items);
				//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
			}
			__syncthreads();
			for(int i = 0; i < n_items; i++) {
				//printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
				if(queue_index + thread_data + i >= *d_size) {
					printf("GPU: exceeded size: %d %d %d %d\n", queue_index, thread_data, i, *d_size);
					return 0;
				}
				d_queue[queue_index + thread_data + i] = items[i];
			}
			return total_items;
		}

	__device__ int pop_id(int id, int &item) {
		if(id < *d_index) {
			//item = cub::ThreadLoad<cub::LOAD_CG>(d_queue + id);
			item = d_queue[id];
			return 1;
		}
		return 0;
	}
};
