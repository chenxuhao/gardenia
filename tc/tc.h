// Copyright (c) 2016, Xuhao Chen

#define TC_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

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


using namespace std;

__global__ void tc_kernel(int m, int *row_offsets, int *column_indices, unsigned *total) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1]; 
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				if(dst > src) break;
				unsigned row_begin_dst = row_offsets[dst];
				unsigned row_end_dst = row_offsets[dst + 1];
				unsigned it = row_begin;
				for (unsigned offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst) {
					int dst_dst = column_indices[offset_dst];
					if(dst_dst > dst) break;
					while(column_indices[it] < dst_dst) it ++;
					if(column_indices[it] == dst_dst) *total ++;
				}
			}
		}
	}
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

void PrintTriangleStats(size_t total_triangles) {
	cout << total_triangles << " triangles" << endl;
}

// uses heuristic to see if worth relabeling
size_t Hybrid(int m, int nnz, int *row_offsets, int *column_indices, int *degree) {
	unsigned h_total = 0;
	unsigned *d_total;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(unsigned), cudaMemcpyHostToDevice));
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	const size_t max_blocks = maximum_residency(tc_kernel, nthreads, 0);
	//if(nblocks > nSM*max_blocks) nblocks = nSM*max_blocks;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	//if (WorthRelabelling(m, row_offsets, column_indices, degree))
	//	OrderedCount<<<nblocks, nthreads>>>(m, row_offsets, column_indices, d_total);
	//else
	tc_kernel<<<nblocks, nthreads>>>(m, row_offsets, column_indices, d_total);
	CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(unsigned), cudaMemcpyDeviceToHost));
	PrintTriangleStats(h_total);
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

