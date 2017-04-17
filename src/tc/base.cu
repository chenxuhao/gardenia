// Copyright (c) 2016, Xuhao Chen
#define TC_VARIANT "topology"
#include <iostream>
#include <cub/cub.cuh>
#include "tc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;
/*
Gardenia Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Xuhao Chen

Will count the number of triangles (cliques of size 3)

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors to be sorted.

Another optimization this implementation has is to relabel the vertices by
degree. This is beneficial if the average degree is high enough and if the
degree distribution is sufficiently non-uniform. To decide whether or not
to relabel the graph, we use the heuristic in WorthRelabelling.
*/

__global__ void tc_kernel(int m, int *row_offsets, int *column_indices, int *total) {
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	int local_total = 0;
	if (src < m) {
		//if (src == 2) printf("src=2\n");
		int row_begin_src = row_offsets[src];
		int row_end_src = row_offsets[src + 1]; 
		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src) {
			int dst = column_indices[offset_src];
			//if (src == 2) printf("\tdst=%d\n", dst);
			if (dst > src) break;
			int row_begin_dst = row_offsets[dst];
			int row_end_dst = row_offsets[dst + 1];
			int it = row_begin_src;
			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
				int dst_dst = column_indices[offset_dst];
				//if (src == 2) printf("\t\tdst_dst=%d\n", dst_dst);
				if (dst_dst > dst) break;
				//if (dst_dst == src) continue;
				//while(column_indices[it] < dst_dst && it <row_end_src) it ++;
				while(column_indices[it] < dst_dst) it ++;
				if(column_indices[it] == dst_dst) local_total += 1;
				/*
				for (it = row_begin_src; it < row_end_src; it ++) {
					int dst_src = column_indices[it];
					if (src == 2) printf("\t\t\tdst_src=%d\n", dst_src);
					if (dst_src == dst_dst) (*total) ++;
					if (dst_src > dst_dst) break;
				}
				*/
			}
		}
	}
	int block_total = BlockReduce(temp_storage).Sum(local_total);
	if(threadIdx.x == 0) atomicAdd(total, block_total);
}

// heuristic to see if sufficently dense power-law graph
bool WorthRelabelling(int m, int nnz, int *row_offsets, int *column_indices, int *degree) {
	int64_t average_degree = nnz / m;
	if (average_degree < 10)
		return false;
	unsigned sp = 1;
	int64_t num_samples = min(int64_t(1000), int64_t(m));
	int64_t sample_total = 0;
	vector<int64_t> samples(num_samples);
	for (int64_t trial=0; trial < num_samples; trial++) {
		samples[trial] = degree[sp++];
		sample_total += samples[trial];
	}
	//sort(samples.begin(), samples.end());
	double sample_average = static_cast<double>(sample_total) / num_samples;
	double sample_median = samples[num_samples/2];
	return sample_average / 2 > sample_median;
}

// uses heuristic to see if worth relabeling
void TCSolver(int m, int nnz, int *h_row_offsets, int *h_column_indices, int *h_degree, int *h_total) {
	Timer t;
	int zero = 0;
	int *d_row_offsets, *d_column_indices;//, *d_degree;
	int *d_total;
	print_device_info(0);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(int)));

	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, m * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_total, &zero, sizeof(int), cudaMemcpyHostToDevice));

	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	int max_blocks = maximum_residency(tc_kernel, nthreads, 0);
	printf("Solving, max_blocks_per_SM=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	t.Start();
	//if (WorthRelabelling(m, nnz, row_offsets, column_indices, degree))
	//	tc_kernel<<<nblocks, nthreads>>>(m, row_offsets, column_indices, total);
	//else
	tc_kernel<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_total);
	CudaTest("solving failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", TC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
	//cout << *h_total << " triangles" << endl;
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_total));
	//CUDA_SAFE_CALL(cudaFree(d_degree));
}

