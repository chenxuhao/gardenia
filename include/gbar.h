#pragma once
#include <cub/cub.cuh>
#include "cutil_subset.h"

// Manages device storage needed for implementing a global software barrier between CTAs in a single grid
class GlobalBarrier {
	public:
		typedef unsigned int SyncFlag;

	protected :
		// Counters in global device memory
		SyncFlag* d_sync;

		__device__ __forceinline__ SyncFlag LoadCG(SyncFlag* d_ptr) const {
			SyncFlag retval;
			retval = cub::ThreadLoad<cub::LOAD_CG>(d_ptr);
			return retval;
		}

	public:
		GlobalBarrier() : d_sync(NULL) {}

		// Synchronize
		__device__ __forceinline__ void Sync() const {
			volatile SyncFlag *d_vol_sync = d_sync;

			// Threadfence and syncthreads to make sure global writes are visible before
			// thread-0 reports in with its sync counter
			__threadfence();
			__syncthreads();

			if (blockIdx.x == 0) {
				// Report in ourselves
				if (threadIdx.x == 0) {
					d_vol_sync[blockIdx.x] = 1;
				}
				__syncthreads();

				// Wait for everyone else to report in
				for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
					//while (d_sync[peer_block] == 0) {
					while (LoadCG(d_sync + peer_block) == 0) {
						__threadfence_block();
					}
				}
				__syncthreads();

				// Let everyone know it's safe to read their prefix sums
				for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
					d_vol_sync[peer_block] = 0;
				}
			} else {
				if (threadIdx.x == 0) {
					// Report in
					d_vol_sync[blockIdx.x] = 1;

					// Wait for acknowledgement
					//while (d_sync[blockIdx.x] == 1) {
					while (LoadCG(d_sync + blockIdx.x) == 1) {
						__threadfence_block();
					}
				}
				__syncthreads();
			}
		}
};

/**
 * Version of global barrier with storage lifetime management.
 * We can use this in host enactors, and pass the base GlobalBarrier
 * as parameters to kernels.
*/
class GlobalBarrierLifetime : public GlobalBarrier {
	protected:
		// Number of bytes backed by d_sync
		size_t sync_bytes;

	public:
		GlobalBarrierLifetime() : GlobalBarrier(), sync_bytes(0) {}

		// Deallocates and resets the progress counters
		cudaError_t HostReset() {
			cudaError_t retval = cudaSuccess;
			if (d_sync) {
				CUDA_SAFE_CALL(cudaFree(d_sync));
				d_sync = NULL;
			}
			sync_bytes = 0;
			return retval;
		}

		virtual ~GlobalBarrierLifetime() {
			HostReset();
		}

		// Sets up the progress counters for the next kernel launch (lazily
		// allocating and initializing them if necessary)
		cudaError_t Setup(int sweep_grid_size) {
			cudaError_t retval = cudaSuccess;
			do {
				size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
				if (new_sync_bytes > sync_bytes) {
					if (d_sync) {
						CUDA_SAFE_CALL(cudaFree(d_sync));
						retval = cudaSuccess;
					}
					sync_bytes = new_sync_bytes;
					CUDA_SAFE_CALL(cudaMalloc((void**) &d_sync, sync_bytes));
					retval = cudaSuccess;
					// Initialize to zero
					CUDA_SAFE_CALL(cudaMemset(d_sync, 0, sweep_grid_size * sizeof(SyncFlag)));
				}
			} while (0);
			return retval;
		}
};
